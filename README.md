# Amazon Product Sentiment Analysis
## Introduction
This project explores **sentiment analysis** on Amazon product reviews using **balanced** and **sample datasets** to investigate the influence of data distributions on model performance. Sentiment analysis involves classifying customer reviews into **positive**, **neutral**, and **negative** categories.

### Key Highlights:
**⚖️ Balanced Dataset**: Contains equal distributions of sentiments, enabling unbiased comparisons of model accuracy across sentiment types.<br>
**📊 Sample Dataset**: Reflects real-world distributions, with a higher prevalence of positive reviews, providing insights into performance under naturally imbalanced conditions.<br>
**📍 Objective**: Assess and compare the strengths and limitations of models in handling skewed data distributions, particularly for neutral sentiments.<br>

## Libraries Used
Data Manipulation: `pandas`, `numpy`
Visualization: `matplotlib`, `seaborn`
Natural Language Processing: `nltk`, `transformers`

## Data Preprocessing
The dataset is derived from Amazon product reviews, containing 568,454 records across 10 columns. Key preprocessing steps include:

1. Dropping irrelevant columns (`ProductId`, `UserId`, etc.) to retain only essential fields: `Score`, `Text`, and an additional label column (`TrueLabel`).<br>
2. The TrueLabel column is created by mapping review scores into sentiment labels:
* Scores 1–2 → Negative
* Score 3 → Neutral
* Scores 4–5 → Positive
```python
df['TrueLabel'] = 'neutral'

df.loc[df['Score'].isin([1, 2]), 'TrueLabel'] = 'negative'

df.loc[df['Score'].isin([4, 5]), 'TrueLabel'] = 'positive'
```
![Count of Review Scores](images/score_vs_sentiment.png)

<p align="center">
  <img src="images/pie_chart.png" alt="Image" width="400">
</p>

The pie chart above shows that 78.1% of the reviews are classified as positive, while the remaining 21.9% are split between negative and neutral sentiments. The bar chart on the left further supports this, with the 5-star bar having the highest count.<br><br>
With such a large proportion of reviews being positive, there may be an inherent bias in the overall sentiment distribution. This bias could skew analysis or decision-making processes, as the majority sentiment could overshadow the experiences of those who have negative or neutral reviews.<br><br>
This sentiment imbalance could lead to overestimation of the overall satisfaction or effectiveness of the product or service, especially if the positive reviews are overrepresented in the analysis. For example, customer satisfaction metrics may be inflated, masking underlying issues that are reflected in negative or neutral reviews.<br><br>


3. Filtering reviews exceeding 512 tokens, due to RoBERTa model constraints. 6005 reviews were removed.<br>

```python
def filter_tokens(df, text_column, model_name='cardiffnlp/twitter-roberta-base-sentiment', token_limit=512):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def token_count(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)  # Add special tokens for the model
        return len(tokens)

    df['token_count'] = df[text_column].apply(token_count)
    filtered_df = df[df['token_count'] <= token_limit]

    filtered_df = filtered_df.drop(columns=['token_count'])

    return filtered_df

filtered_df = filter_tokens(df, 'Text')
```
    
4. Generating two datasets:
* Balanced Dataset: Equal samples of all sentiments.
* Sample Dataset: Randomly sampled records to mimic real-world distribution.<br>

![Count of Review Scores (Sample)](images/balance_sample_charts.png)

The Balanced Dataset and the Sample Dataset each contain 1500 reviews.<br>

**Balanced Dataset**<br>
The balanced dataset includes 250 reviews each for 1-, 2-, 4-, and 5-star ratings, along with 500 reviews for 3-star ratings.
```python
df1 = filtered_df[filtered_df['Score']==1].sample(250, random_state=42)
df2 = filtered_df[filtered_df['Score']==2].sample(250, random_state=42)
df3 = filtered_df[filtered_df['Score']==3].sample(500, random_state=42)
df4 = filtered_df[filtered_df['Score']==4].sample(250, random_state=42)
df5 = filtered_df[filtered_df['Score']==5].sample(250, random_state=42)

df_balance = pd.concat([df1, df2, df3, df4, df5])
```

**Sample Dataset**<br>
The sample dataset contains a random sample of 1500 reviews from the filtered dataset. <br>
```python
df_sample = filtered_df.sample(1500, random_state=42)
```



