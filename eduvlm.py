import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer,
    CLIPVisionModel, CLIPProcessor,
)
from sentence_transformers import SentenceTransformer
from PIL import Image
import json
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict
import os
from sklearn.metrics import mean_absolute_error
import torchmetrics.functional as tmf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AssessmentConfig:
    vision_model = "openai/clip-vit-base-patch32"
    language_model = SentenceTransformer('all-MiniLM-L6-v2')
    hidden_dim = 512
    num_assessment_criteria = 5
    max_text_length = 512
    image_size = 264
    learning_rate: float = 2e-5
    batch_size = 8
    num_epochs = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EducationalAssessmentDataset(Dataset):
    def __init__(self, data_file: str, processor, tokenizer, config: AssessmentConfig):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if not os.path.exists(item['image_path']):
            raise FileNotFoundError(f"Image not found: {item['image_path']}")

        image = Image.open(item['image_path']).convert('RGB')
        image_inputs = self.processor(images=image, return_tensors="pt")

        text = item['text_content']
        text_inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_text_length,
            return_tensors="pt"
        )

        scores = torch.tensor(item['scores'], dtype=torch.float32)

        return {
            'image': image_inputs['pixel_values'].squeeze(0),
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'scores': scores,
            'assignment_type': item.get('assignment_type', 'general')
        }

class VisionLanguageFusion(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, text_features):
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)

        combined = torch.cat([vision_proj.unsqueeze(1), text_proj.unsqueeze(1)], dim=1)
        fused, _ = self.fusion(combined, combined, combined)
        fused = self.layer_norm(fused)

        return fused.mean(dim=1)

class EducationalVLM(nn.Module):
    def __init__(self, config: AssessmentConfig):
        super().__init__()
        self.config = config

        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_model)
        vision_dim = self.vision_model.config.hidden_size

        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        text_dim = self.text_model.config.hidden_size

        self.fusion = VisionLanguageFusion(vision_dim, text_dim, config.hidden_dim)

        self.assessment_heads = nn.ModuleDict({
            'content_accuracy': nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'presentation_quality': nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'analysis_depth': nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'overall_score': nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        })

        self.feedback_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, len(self.get_feedback_vocabulary()))  # No softmax here
        )

    def get_feedback_vocabulary(self):
        return [
            "excellent_content", "good_content", "needs_improvement_content",
            "clear_presentation", "unclear_presentation",
            "thorough_analysis", "superficial_analysis",
            "well_structured", "poorly_structured"
        ]

    def forward(self, image, text_input_ids, text_attention_mask):
        vision_outputs = self.vision_model(pixel_values=image)
        vision_features = vision_outputs.pooler_output

        text_outputs = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.pooler_output

        fused_features = self.fusion(vision_features, text_features)

        scores = {}
        for criterion, head in self.assessment_heads.items():
            scores[criterion] = head(fused_features).squeeze(-1)

        feedback_logits = self.feedback_generator(fused_features)
        return scores, feedback_logits

class AssessmentTrainer:
    def __init__(self, model, config: AssessmentConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        self.mse_loss = nn.MSELoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            image = batch['image'].to(self.device)
            text_ids = batch['text_input_ids'].to(self.device)
            text_mask = batch['text_attention_mask'].to(self.device)
            target_scores = batch['scores'].to(self.device)

            predicted_scores, _ = self.model(image, text_ids, text_mask)

            loss = 0
            for i, criterion in enumerate(self.model.assessment_heads.keys()):
                if i < target_scores.shape[1]:
                    loss += self.mse_loss(predicted_scores[criterion], target_scores[:, i])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(self.device)
                text_ids = batch['text_input_ids'].to(self.device)
                text_mask = batch['text_attention_mask'].to(self.device)
                target_scores = batch['scores'].to(self.device)

                predicted_scores, _ = self.model(image, text_ids, text_mask)
                loss = 0
                batch_predictions = []
                for i, criterion in enumerate(self.model.assessment_heads.keys()):
                    if i < target_scores.shape[1]:
                        criterion_loss = self.mse_loss(predicted_scores[criterion], target_scores[:, i])
                        loss += criterion_loss
                        batch_predictions.append(predicted_scores[criterion].cpu().numpy())

                total_loss += loss.item()
                all_predictions.extend(np.array(batch_predictions).T.tolist())
                all_targets.extend(target_scores.cpu().numpy().tolist())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        correlation = tmf.pearson_corrcoef(torch.tensor(targets.flatten()), torch.tensor(predictions.flatten())).item()

        return {
            'val_loss': total_loss / len(val_loader),
            'mae': mae,
            'correlation': correlation
        }

    def train(self, train_loader, val_loader):
        best_correlation = -1
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"MAE: {val_metrics['mae']:.4f}")
            logger.info(f"Correlation: {val_metrics['correlation']:.4f}")

            if val_metrics['correlation'] > best_correlation:
                best_correlation = val_metrics['correlation']
                torch.save(self.model.state_dict(), 'eduvlm.pth')
                logger.info("Saved new best model!")

class AssessmentInference:
    def __init__(self, model_path: str, config: AssessmentConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.model = EducationalVLM(config)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(config.vision_model)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def assess_submission(self, image_path: str, text_content: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image_inputs = self.processor(images=image, return_tensors="pt")

        text_inputs = self.tokenizer(
            text_content,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_text_length,
            return_tensors="pt"
        )

        image_tensor = image_inputs['pixel_values'].to(self.device)
        text_ids = text_inputs['input_ids'].to(self.device)
        text_mask = text_inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            scores, feedback_logits = self.model(image_tensor, text_ids, text_mask)

            results = {criterion: float(score_tensor.cpu().item()) for criterion, score_tensor in scores.items()}
            feedback_probs = torch.softmax(feedback_logits, dim=-1)
            top_feedback = torch.topk(feedback_probs, k=3, dim=-1)

            feedback_vocab = self.model.get_feedback_vocabulary()
            results['feedback'] = [
                (feedback_vocab[idx.item()], prob.item())
                for idx, prob in zip(top_feedback.indices[0], top_feedback.values[0])
            ]

        return results

def create_sample_data():
    sample_data = [
        {
            "image_path": "sample_lab_report1.jpg",
            "text_content": "",
            "scores": [0.85, 0.75, 0.80, 0.78],
            "assignment_type": "lab_report"
        },
        {
            "image_path": "sample_essay1.jpg",
            "text_content": "",
            "scores": [0.70, 0.85, 0.65, 0.73],
            "assignment_type": "essay"
        },
        {
            "image_path": "sample_programming_assignment1.jpg",
            "text_content": "",
            "scores": [0.70, 0.85, 0.65, 0.73],
            "assignment_type": "essay"
        },
    ]

    with open('training_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)

def main():
    config = AssessmentConfig()
    create_sample_data()

    model = EducationalVLM(config)
    processor = CLIPProcessor.from_pretrained(config.vision_model)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = EducationalAssessmentDataset('training_data.json', processor, tokenizer, config)

    if os.path.exists('validation_data.json'):
        val_dataset = EducationalAssessmentDataset('validation_data.json', processor, tokenizer, config)
    else:
        logger.warning("Validation data not found. Using training data as validation (not recommended).")
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    trainer = AssessmentTrainer(model, config)
    trainer.train(train_loader, val_loader)

    inference = AssessmentInference('eduvlm.pth', config)
    results = inference.assess_submission('new_submission.jpg', 'Student name here...')
    print("Assessment Results:", results)

if __name__ == "__main__":
    main()
