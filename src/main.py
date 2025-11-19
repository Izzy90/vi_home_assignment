from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
import xgboost
from sentence_transformers import SentenceTransformer
import simple_icd_10 as icd

from .config import DATA_PATH


## Setup
train_web_visits = pd.read_csv(f'{DATA_PATH}/train/web_visits.csv')
train_claims_df = pd.read_csv(f'{DATA_PATH}/train/claims.csv')
train_churn_labels_df = pd.read_csv(f'{DATA_PATH}/train/churn_labels.csv')
train_app_usage_df = pd.read_csv(f'{DATA_PATH}/train/app_usage.csv')

test_web_visits_df = pd.read_csv(f'{DATA_PATH}/test/test_web_visits.csv')
test_claims_df = pd.read_csv(f'{DATA_PATH}/test/test_claims.csv')
test_churn_labels_df = pd.read_csv(f'{DATA_PATH}/test/test_churn_labels.csv')
test_app_usage_df = pd.read_csv(f'{DATA_PATH}/test/test_app_usage.csv')

# filter-out members that had outreach, to avoid using this effect
train_churn_labels_df = train_churn_labels_df[train_churn_labels_df['outreach'] == 0]

## Feature Engineering
train_app_usage_df['timestamp'] = pd.to_datetime(train_app_usage_df['timestamp'])
train_web_visits['timestamp'] = pd.to_datetime(train_web_visits['timestamp'])
train_claims_df['diagnosis_date'] = pd.to_datetime(train_claims_df['diagnosis_date'])
train_churn_labels_df['signup_date'] = pd.to_datetime(train_churn_labels_df['signup_date'])

app_window_end = train_app_usage_df['timestamp'].max()
web_window_end = train_web_visits['timestamp'].max()
claims_window_end = train_claims_df['diagnosis_date'].max()
observation_window_end = max(app_window_end, web_window_end, claims_window_end)
observation_window_end

### App usage feature candidates

train_app_usage_df['session_date'] = train_app_usage_df['timestamp'].dt.date
app_features = (
    train_app_usage_df
    .groupby('member_id')
    .agg(
        session_count=('timestamp', 'size'),
        active_session_days=('session_date', 'nunique'),
        first_session_ts=('timestamp', 'min'),
        last_session_ts=('timestamp', 'max')
    )
)
app_features['session_span_days'] = (app_features['last_session_ts'] - app_features['first_session_ts']).dt.days.clip(lower=0) + 1
app_features['sessions_per_active_day'] = app_features['session_count'] / app_features['active_session_days'].replace(0, pd.NA)
app_features['days_since_last_session'] = (observation_window_end - app_features['last_session_ts']).dt.days

### Web visit feature candidates

train_web_visits['topic'] = train_web_visits['url'].apply(lambda x: x.split(".")[1].split('/')[1])
train_web_visits['visit_date'] = train_web_visits['timestamp'].dt.date
web_features = (
    train_web_visits
    .groupby('member_id')
    .agg(
        web_visit_count=('timestamp', 'size'),
        unique_urls=('url', 'nunique'),
        unique_titles=('title', 'nunique'),
        web_active_days=('visit_date', 'nunique'),
        first_web_visit=('timestamp', 'min'),
        last_web_visit=('timestamp', 'max'),
        main_interest=('topic', lambda x: x.value_counts().idxmax())
    ).reset_index()
)
web_features['days_since_last_web_visit'] = (observation_window_end - web_features['last_web_visit']).dt.days
web_features['visits_per_web_day'] = web_features['web_visit_count'] / web_features['web_active_days'].replace(0, pd.NA)

web_visits_topic_counts = train_web_visits.pivot_table(
    index='member_id', 
    columns='topic', 
    values='url', 
    aggfunc='count', 
    fill_value=0
).reset_index()

web_visits_topic_counts.columns.name = None

# Calculate the ratio for each topic out of all topic counts per member_id
topic_cols = [col for col in web_visits_topic_counts.columns if col != 'member_id']
web_visits_topic_counts_with_ratios = web_visits_topic_counts.copy()
row_sums = web_visits_topic_counts_with_ratios[topic_cols].sum(axis=1)
for col in topic_cols:
    web_visits_topic_counts_with_ratios[f'{col}_ratio'] = (
        web_visits_topic_counts_with_ratios[col] / row_sums
    )

web_features = web_features.merge(web_visits_topic_counts_with_ratios, on='member_id', how='left')


# # Find the most common topic for each member_id
# main_interest_map = (
#     train_web_visits
#     .groupby('member_id')['topic']
#     .agg(lambda x: x.value_counts().idxmax())
# )
# # Map this value back to each row in train_web_visits
# train_web_visits['main_interest'] = train_web_visits['member_id'].map(main_interest_map)

# interests = list(train_web_visits['topic'].unique())
# try: 
#     embedding_model
# except NameError:
#     embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# # Get embeddings for each topic in interests
# interest_embeddings = embedding_model.encode(interests, show_progress_bar=True, truncate_dim=128)

# # Create a DataFrame with each topic and its embedding vector
# interest_embeddings_df = pd.DataFrame({
#     'topic': interests,
#     'embedding': list(interest_embeddings)
# })

# # Explode the embedding column into multiple columns (one per embedding dimension)
# embedding_dims = len(interest_embeddings_df.iloc[0]['embedding'])
# embedding_cols = [f'interest_embedding_dim_{i}' for i in range(embedding_dims)]
# embedding_df_expanded = pd.DataFrame(interest_embeddings_df['embedding'].tolist(), columns=embedding_cols)
# interest_embeddings_exploded_df = pd.concat([interest_embeddings_df.drop(columns=['embedding']), embedding_df_expanded], axis=1)

# web_features = web_features.merge(interest_embeddings_exploded_df, left_on='main_interest', right_on='topic')

### Claims-based feature candidates

train_claims_df['icd_chapter'] = train_claims_df['icd_code'].str[0]
claims_features = (
    train_claims_df
    .groupby('member_id')
    .agg(
        claim_count=('diagnosis_date', 'size'),
        unique_icd_codes=('icd_code', 'nunique'),
        last_claim_date=('diagnosis_date', 'max')
    )
)
claims_features['days_since_last_claim'] = (observation_window_end - claims_features['last_claim_date']).dt.days
chapters = train_claims_df['icd_chapter'].value_counts().index.tolist()
chapter_pivot = (
    train_claims_df[train_claims_df['icd_chapter'].isin(chapters)]
    .pivot_table(index='member_id', columns='icd_chapter', values='icd_code', aggfunc='count', fill_value=0)
)
chapter_pivot.columns = [f"claim_chapter_{col}" for col in chapter_pivot.columns]
claims_features = claims_features.join(chapter_pivot, how='left').fillna(0)

# Make a row per member-icd_code
member_icd_counts = (
    train_claims_df.groupby(['member_id', 'icd_code'])
    .size()
    .reset_index(name='count')
)

# Calculate total claims per member
member_total_claims = (
    train_claims_df.groupby('member_id')
    .size()
    .rename('total_claims')
    .reset_index()
)

# Merge counts with total claims to get the ratio
member_icd_counts = member_icd_counts.merge(member_total_claims, on='member_id', how='left')
member_icd_counts['ratio'] = member_icd_counts['count'] / member_icd_counts['total_claims']

# Pivot such that for each member, for every icd_code there are two columns:
# 'icd_{code}_count' and 'icd_{code}_ratio'
icd_code_list = sorted(train_claims_df['icd_code'].unique())
# Build pivot tables for counts and ratios
counts_pivot = member_icd_counts.pivot(index='member_id', columns='icd_code', values='count')
ratios_pivot = member_icd_counts.pivot(index='member_id', columns='icd_code', values='ratio')

# Rename columns accordingly
counts_pivot.columns = [f'icd_{col}_count' for col in counts_pivot.columns]
ratios_pivot.columns = [f'icd_{col}_ratio' for col in ratios_pivot.columns]

# Concatenate both count and ratio columns
member_icd_counts_ratios_df = pd.concat([counts_pivot, ratios_pivot], axis=1).reset_index().fillna(0)

# add to claims_features
claims_features = claims_features.merge(member_icd_counts_ratios_df, on='member_id', how='left')

### Combine engineered features with churn labels

feature_df = (
    train_churn_labels_df[['member_id', 'signup_date', 'churn', 'outreach']]
    .merge(app_features, on='member_id', how='left')
    .merge(web_features, on='member_id', how='left')
    .merge(claims_features, on='member_id', how='left')
)
feature_df['member_tenure_days'] = (observation_window_end - feature_df['signup_date']).dt.days
numeric_cols = [col for col in feature_df.select_dtypes(include=['number']).columns if col not in ['churn']]
feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)

train_session_count_by_member_df = train_app_usage_df[['member_id', 'timestamp']].groupby('member_id') \
                                                                                 .agg('count') \
                                                                                 .rename({'timestamp': 'session_count'}, axis=1)
train_data_df = train_churn_labels_df.merge(train_session_count_by_member_df, on='member_id', how='left')

feature_df = feature_df.select_dtypes(include=['number'])
feature_df = feature_df.drop('churn', axis=1)
train_data_df = train_data_df.merge(feature_df, on='member_id', how='left')

print("train_data_df:")
pprint(train_data_df.head(10))


## Data preparation
X = train_data_df.drop(columns=["churn"])
y = train_data_df["churn"]

X.drop('signup_date', axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

## Model
# initialize the xgboost classifier
clf = xgboost.XGBClassifier(
    n_estimators=100
)

## fit the model
clf.fit(X_train, y_train)

## make predictions
y_pred = clf.predict_proba(X_val)[:, 1]

## evaluate 


# print("Confusion Matrix:")
# print(confusion_matrix(y_val, y_pred))

roc_auc = roc_auc_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred)
# precision_s, recall_s, f1_s, support = precision_recall_fscore_support(y_val, y_pred)

print(f"ROC-AUC: {roc_auc:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print("Support:", support)




print("Reached End")