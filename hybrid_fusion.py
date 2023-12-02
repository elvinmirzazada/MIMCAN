# Created by elvinmirzazada at 23:11 25/05/2023 using PyCharm

class HybridFusionModel(nn.Module):
    def __init__(self, bert, resnet):
        super().__init__()
        self.bert = bert
        self.resnet = resnet
        self.drop = nn.Dropout(p=0.3)

        # Early Fusion
        self.image_to_title_attention = nn.MultiheadAttention(bert.config.hidden_size, num_heads=4)  # Cross-attention

        # Late Fusion
        self.linear = nn.Linear(1536, bert.config.hidden_size)
        self.norm = nn.BatchNorm1d(bert.config.hidden_size)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(bert.config.hidden_size * 2, bert.config.hidden_size)  # Modify the hidden layer input size
        self.classifier = nn.Linear(bert.config.hidden_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, inputs, images):
        # Process text input
        text_output = self.bert(**inputs).last_hidden_state[:, 0, :]

        # Early Fusion
        scores = []
        attention_outputs = []
        for img in images:
            img_emb = self.resnet(img).view(-1)
            img_emb = self.linear(img_emb)
            img_emb = self.norm(img_emb)
            img_emb = self.relu(img_emb)

            img_emb = img_emb.view(1, 1, -1)
            att_out, score = self.image_to_title_attention(text_output.unsqueeze(1), img_emb, img_emb)
            score = [s.cpu() for s in score]
            scores.append(score)
            attention_outputs.append(att_out)

        # Late Fusion
        attention_output = torch.stack(attention_outputs).mean(dim=0)
        fused_output = torch.cat((text_output, attention_output.squeeze(1)), dim=1)

        # Classifier
        logits = self.hidden(fused_output)
        logits = self.drop(logits)
        logits = self.classifier(logits)

        return self.softmax(logits), scores
