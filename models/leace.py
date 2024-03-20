from torch import nn
import numpy as np
from sklearn.linear_model import SGDClassifier
import sklearn
from .dpr import mDPRBase
from concept_erasure import LeaceFitter

EVAL_CLF_PARAMS = {"loss": "log_loss", "tol": 1e-4, "iters_no_change": 15, "alpha": 1e-4, "max_iter": 25000}
NUM_CLFS_IN_EVAL = 1 # change to 1 for large dataset / high dimensionality

def init_classifier():

    return SGDClassifier(loss=EVAL_CLF_PARAMS["loss"], fit_intercept=True, max_iter=EVAL_CLF_PARAMS["max_iter"], tol=EVAL_CLF_PARAMS["tol"], n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
                        n_jobs=32, alpha=EVAL_CLF_PARAMS["alpha"])
                        

def get_score(X_train, y_train, X_dev, y_dev):
    loss_vals = []
    train_accs = []
    dev_accs = []
    
    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_dev)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        train_accs.append(clf.score(X_train, y_train))
        dev_accs.append(clf.score(X_dev, y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], train_accs[i], dev_accs[i]


class mDPRScrubber(mDPRBase):
    def post_process(self, encoder_output, fitter: LeaceFitter=None):
        output = nn.functional.normalize(encoder_output, dim=-1) * self.temperature if self.normalize else encoder_output * self.temperature
        if fitter is not None:
            output = fitter.eraser(output.float())
        return output

    def query(self, input_ids, attention_mask, fitter: LeaceFitter=None):
        enc_output = self.feature(input_ids, attention_mask=attention_mask, return_pooler=self.use_pooler)
        return self.post_process(enc_output, fitter)

    def doc(self, input_ids, attention_mask, fitter: LeaceFitter=None):
        return self.query(input_ids, attention_mask, fitter)

    def match(self, q_ids, q_mask, d_ids, d_mask, fitter: LeaceFitter=None):
        q_reps = self.query(q_ids, q_mask, fitter)
        d_reps = self.doc(d_ids, d_mask, fitter)
        scores = self.score(q_reps, d_reps)
        loss = self.loss_fct(scores, self.labels[:scores.size(0)])
        return loss

    def forward(self, q_ids, q_mask, d_ids, d_mask, fitter: LeaceFitter=None):
        return self.match(q_ids, q_mask, d_ids, d_mask, fitter)