
import torch
import wandb
from transformers import Trainer

class ORPOTrainer(Trainer):
    def __init__(self, alpha, pad, disable_prompt_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = pad
        # Make alpha a learnable parameter initialized at 0.5
        self.alpha = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Hybrid approach: learnable weight for token vs sequence level
        # beta=0 means pure sequence-level, beta=1 means pure token-level
        self.beta = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.disable_prompt_loss = disable_prompt_loss
        print("Pad Token ID: ", self.pad)
        print(f"Learnable alpha initialized: {self.alpha.item()}")
        print(f"Learnable beta (hybrid weight) initialized: {self.beta.item()}")
    
    def get_optimizer_grouped_parameters(self):
        """
        Override to include the learnable alpha and beta parameters in the optimizer.
        """
        # Get model parameters with weight decay
        model_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0.01,
            }
        ]
        
        # Add alpha and beta parameters without weight decay
        model_params.append({
            "params": [self.alpha, self.beta],
            "weight_decay": 0.0,
            "lr": self.args.learning_rate if hasattr(self.args, 'learning_rate') else 1e-5,
        })
        
        return model_params
    
    def create_optimizer(self):
        """Override to ensure alpha and beta are included in optimizer."""
        if self.optimizer is None:
            optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
            
            from torch.optim import AdamW
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate if hasattr(self.args, 'learning_rate') else 1e-5,
                betas=(self.args.adam_beta1 if hasattr(self.args, 'adam_beta1') else 0.9,
                       self.args.adam_beta2 if hasattr(self.args, 'adam_beta2') else 0.999),
                weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0.01,
            )
        
        return self.optimizer
        
    def compute_custom_loss(self, logits, labels):
        
        logits = logits.contiguous()
        
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.transpose(2, 1), shift_labels).mean(dim=-1)
            
        return loss
    
    def compute_logps(self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
        mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2, 
                                       index=(mask * chosen_inputs[:, 1:]).unsqueeze(2).to(torch.int64)).squeeze(2)
        # Apply mask
        per_token_logps = per_token_logps * mask.to(dtype=per_token_logps.dtype)
        mask_sum = mask.sum(dim=1).clamp(min=1)
        result = per_token_logps.sum(dim=1) / mask_sum.to(dtype=per_token_logps.dtype)
        # Replace any NaN/Inf with 0
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result
    
    def compute_token_level_logps(self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
        """
        Compute per-token log probabilities for the response part (excluding prompt).
        Returns: (per_token_logps, mask) where mask indicates response tokens.
        """
        # mask = 1 for response tokens, 0 for prompt tokens
        mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        mask_float = mask.to(dtype=torch.bfloat16)
        
        # Get log probabilities for each token (index must be int64)
        per_token_logps = torch.gather(
            logits[:, :-1, :].log_softmax(-1), 
            dim=2, 
            index=(mask * chosen_inputs[:, 1:]).unsqueeze(2).to(torch.int64)
        ).squeeze(2)
        
        # Apply mask
        per_token_logps = per_token_logps * mask_float
        # Replace any NaN/Inf with 0
        per_token_logps = torch.where(torch.isfinite(per_token_logps), per_token_logps, torch.zeros_like(per_token_logps))
        
        return per_token_logps, mask_float
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # Generate the hidden states for 'chosen' and 'reject'
        neg_labels = inputs['negative_input_ids'].clone()
        pos_labels = inputs['positive_input_ids'].clone()

        ### Discard the prompt tokens in NLL loss if true
        if self.disable_prompt_loss:
            mask = inputs['attention_mask'] * inputs['positive_attention_mask']
            pos_labels = pos_labels * mask.logical_not()
            pos_labels[pos_labels == 0] = self.pad
        ##################################################

        neg_labels[neg_labels == self.pad] = -100
        pos_labels[pos_labels == self.pad] = -100

        

        outputs_neg = model(**{'input_ids': inputs['negative_input_ids'],
                               'attention_mask': inputs['negative_attention_mask'],
                               'labels': neg_labels,}, output_hidden_states=True)      
        outputs_pos = model(**{'input_ids': inputs['positive_input_ids'],
                               'attention_mask': inputs['positive_attention_mask'],
                               'labels': pos_labels,}, output_hidden_states=True)
            
        # Calculate NLL loss
        pos_loss = outputs_pos.loss
        
        # =============================================
        # HYBRID APPROACH: Token-level + Sequence-level
        # =============================================
        
        # 1. Sequence-level log probabilities (original ORPO)
        pos_seq_prob = self.compute_logps(
            prompt_attention_mask=inputs['attention_mask'], 
            chosen_inputs=inputs['positive_input_ids'], 
            chosen_attention_mask=inputs['positive_attention_mask'], 
            logits=outputs_pos.logits
        )
        neg_seq_prob = self.compute_logps(
            prompt_attention_mask=inputs['attention_mask'], 
            chosen_inputs=inputs['negative_input_ids'], 
            chosen_attention_mask=inputs['negative_attention_mask'], 
            logits=outputs_neg.logits
        )
        
        # 2. Token-level log probabilities
        pos_token_logps, pos_mask = self.compute_token_level_logps(
            prompt_attention_mask=inputs['attention_mask'],
            chosen_inputs=inputs['positive_input_ids'],
            chosen_attention_mask=inputs['positive_attention_mask'],
            logits=outputs_pos.logits
        )
        neg_token_logps, neg_mask = self.compute_token_level_logps(
            prompt_attention_mask=inputs['attention_mask'],
            chosen_inputs=inputs['negative_input_ids'],
            chosen_attention_mask=inputs['negative_attention_mask'],
            logits=outputs_neg.logits
        )
        
        # 3. Compute token-level log odds ratio for each token position
        # Per-token log odds: log(p_token / (1 - p_token))
        # Use soft clamp for numerical stability
        pos_token_log_odds = pos_token_logps - torch.log1p(-torch.exp(pos_token_logps).clamp(max=1-1e-7))
        neg_token_log_odds = neg_token_logps - torch.log1p(-torch.exp(neg_token_logps).clamp(max=1-1e-7))
        
        # Token-level ratio (sum across response tokens)
        pos_mask_sum = pos_mask.sum(dim=1).clamp(min=1)
        neg_mask_sum = neg_mask.sum(dim=1).clamp(min=1)
        pos_token_ratio = (pos_token_log_odds * pos_mask).sum(dim=1) / pos_mask_sum
        neg_token_ratio = (neg_token_log_odds * neg_mask).sum(dim=1) / neg_mask_sum
        
        # 4. Hybrid combination using learnable beta
        # beta controls the balance: beta=0 -> pure sequence, beta=1 -> pure token
        # Clamp beta to [0, 1] to keep it in valid range
        beta = torch.sigmoid(self.beta)  # Sigmoid to constrain to [0, 1]
        
        # Sequence-level log odds ratio
        # Use soft clamp for numerical stability
        seq_log_odds = (pos_seq_prob - neg_seq_prob) - (
            torch.log1p(-torch.exp(pos_seq_prob).clamp(max=1-1e-7)) - torch.log1p(-torch.exp(neg_seq_prob).clamp(max=1-1e-7))
        )
        
        # Token-level log odds ratio
        token_log_odds = pos_token_ratio - neg_token_ratio
        
        # Hybrid log odds ratio
        hybrid_log_odds = (1 - beta) * seq_log_odds + beta * token_log_odds
        
        # Final clamp to prevent NaN
        hybrid_log_odds = torch.clamp(hybrid_log_odds, min=-100, max=100)
        # Replace any remaining NaN/Inf with 0
        hybrid_log_odds = torch.where(torch.isfinite(hybrid_log_odds), hybrid_log_odds, torch.zeros_like(hybrid_log_odds))
        
        # Calculate final ratio
        sig_ratio = torch.nn.functional.sigmoid(hybrid_log_odds)
        ratio = torch.log(torch.clamp(sig_ratio, min=1e-10, max=1-1e-10))  # Ensure valid log range
        
        # Calculate the Final Loss with alpha
        # Keep in float32 to preserve gradients through alpha
        alpha_loss = torch.mean(pos_loss.float() - self.alpha * ratio.float())
        
        # Only fallback to pos_loss if completely NaN (rare case)
        loss = torch.where(
            torch.isfinite(alpha_loss), 
            alpha_loss, 
            pos_loss.float()
        ).to(dtype=torch.bfloat16)
        
        # Safe logging values
        def safe_mean(tensor):
            return torch.mean(torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))).item()
        
        wandb.log({
            'Positive Geometric Mean': safe_mean(pos_seq_prob),
            'Negative Geometric Mean': safe_mean(neg_seq_prob),
            'Log Odds Ratio': safe_mean(ratio),
            'Log Odds': safe_mean(hybrid_log_odds),
            'Alpha (lambda)': self.alpha.item(),
            'Beta (hybrid weight)': beta.item(),
            'Sequence Log Odds': safe_mean(seq_log_odds),
            'Token Log Odds': safe_mean(token_log_odds),
        })
        
        return (loss, outputs_pos) if return_outputs else loss