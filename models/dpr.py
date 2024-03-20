import torch
from torch import nn
import torch.nn.functional as F

class mDPRBase(nn.Module):
    def __init__(self, base_encoder, args):
        super(mDPRBase, self).__init__()
        self.base_encoder = base_encoder
        self.args = args
        self.is_ance = args.base_model_name == "castorini/ance-msmarco-passage"
        self.labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        self.use_pooler = args.use_pooler
        self.normalize = args.normalize
        self.temperature = args.temperature
        self.loss_fct = nn.CrossEntropyLoss()

    def match(self, q_ids, q_mask, d_ids, d_mask):
        q_reps = self.query(q_ids, q_mask)
        d_reps = self.doc(d_ids, d_mask)
        scores = self.score(q_reps, d_reps)
        loss = self.loss_fct(scores, self.labels[:scores.size(0)])
        return loss
    
    def feature(self, input_ids, attention_mask=None, return_pooler=True):
        enc_reps = self.base_encoder(input_ids, attention_mask=attention_mask)
        if self.is_ance:
            enc_output = enc_reps
        elif return_pooler:
            enc_output = enc_reps.pooler_output
        else:
            enc_output = enc_reps.last_hidden_state[:, 0, :]
        return enc_output
    
    def post_process(self, encoder_output):
        return nn.functional.normalize(encoder_output, dim=-1) * self.temperature if self.normalize else encoder_output * self.temperature
    
    def query(self, input_ids, attention_mask):
        enc_output = self.feature(input_ids, attention_mask=attention_mask, return_pooler=self.use_pooler)
        return self.post_process(enc_output)

    def doc(self, input_ids, attention_mask):
        return self.query(input_ids, attention_mask)

    def score(self, query_reps, document_reps):
        return query_reps.mm(document_reps.t())

    def forward(self, q_ids, q_mask, d_ids, d_mask):
        return self.match(q_ids, q_mask, d_ids, d_mask)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if type(checkpoint) is dict:
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)