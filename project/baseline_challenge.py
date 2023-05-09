# TODO: In this cell, write your BaselineChallenge flow in the baseline_challenge.py file.

from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np 
from dataclasses import dataclass

def labeling_function(row): 
    if row['rating'] >= 4: 
        return 1 
    else: 
        return 0

@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None

class BaselineChallenge(FlowSpec):

    split_size = Parameter('split-sz', default=0.2)
    data = IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv')
    kfold = Parameter('k', default=5)
    scoring = Parameter('scoring', default='accuracy')

    @step
    def start(self):

        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv("/home/workspace/workspaces/full-stack-ml-metaflow-corise-week-1/data/Womens Clothing E-Commerce Reviews.csv")  
        # Look up a few lines to the IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv'). 
        # You can find documentation on IncludeFile here: https://docs.metaflow.org/scaling/data#data-in-local-files


        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df = df[~df.review_text.isna()]
        df['review'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({'label': labels, **_has_review_df})

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')

        self.next(self.baseline, self.model)

    @step
    def baseline(self):
        "Compute the baseline"

        from sklearn.metrics import accuracy_score, roc_auc_score
        self._name = "baseline"
        params = "Always predict 1"
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"
        major_class = self.traindf['label'].mode()[0]
        predictions = [major_class] * len(self.valdf) # TODO: predict the majority class
        acc = accuracy_score(self.valdf['label'], predictions) # TODO: return the accuracy_score of these predictions
        rocauc = roc_auc_score(self.valdf['label'], predictions) # TODO: return the roc_auc_score of these predictions
        self.result = ModelResult("Baseline", params, pathspec, acc, rocauc)
        self.next(self.aggregate)

    @step
    def model(self):

        # TODO: import your model if it is defined in another file.
        from model import NbowModel
        self._name = "model"
        # NOTE: If you followed the link above to find a custom model implementation, 
            # you will have noticed your model's vocab_sz hyperparameter.
            # Too big of vocab_sz causes an error. Can you explain why? 
        self.hyperparam_set = [{'vocab_sz': 100}, {'vocab_sz': 300}, {'vocab_sz': 500}]  
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        self.results = []
        for params in self.hyperparam_set:
            model = NbowModel(vocab_sz=params['vocab_sz']) # TODO: instantiate your custom model here!
            model.fit(X=self.df['review'], y=self.df['label'])
            acc = model.eval_acc(self.valdf['review'].values, self.valdf['label']) # TODO: evaluate your custom model in an equivalent way to accuracy_score.
            rocauc = model.eval_rocauc(self.valdf['review'].values, self.valdf['label']) # TODO: evaluate your custom model in an equivalent way to roc_auc_score.
            self.results.append(ModelResult(f"NbowModel - vocab_sz: {params['vocab_sz']}", params, pathspec, acc, rocauc))

        self.next(self.aggregate)

    def add_one(self, rows, result, df):
        "A helper function to load results."
        rows.append([
            Markdown(result.name),
            Artifact(result.params),
            Artifact(result.pathspec),
            Artifact(result.acc),
            Artifact(result.rocauc)
        ])
        df['name'].append(result.name)
        df['accuracy'].append(result.acc)
        return rows, df

    @card(type='corise') # TODO: Set your card type to "corise". 
            # I wonder what other card types there are?
            # https://docs.metaflow.org/metaflow/visualizing-results
            # https://github.com/outerbounds/metaflow-card-altair/blob/main/altairflow.py
    @step
    def aggregate(self, inputs):

        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import rcParams 
        rcParams.update({'figure.autolayout': True})

        rows = []
        violin_plot_df = {'name': [], 'accuracy': []}
        for task in inputs:
            if task._name == "model": 
                for result in task.results:
                    print(result)
                    rows, violin_plot_df = self.add_one(rows, result, violin_plot_df)
            elif task._name == "baseline":
                print(task.result)
                rows, violin_plot_df = self.add_one(rows, task.result, violin_plot_df)
            else:
                raise ValueError("Unknown task._name type. Cannot parse results.")
            
        current.card.append(Markdown("# All models from this flow run"))

        current.card.append(
            Table(
                rows,  
                headers=["Model name", "Params", "Task pathspec", "Accuracy", "ROCAUC"]
            )
        )
        
        
        fig, ax = plt.subplots(1,1)
        plt.xticks(rotation=40)
        sns.violinplot(data=violin_plot_df, x="name", y="accuracy", ax=ax)
        
        current.card.append(Image.from_matplotlib(fig))
        # Docs: https://docs.metaflow.org/metaflow/visualizing-results/easy-custom-reports-with-card-components#showing-plots

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    BaselineChallenge()
