import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

file_path = 'data/results.csv'
csv_url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'

if not os.path.exists(file_path):
    df_remote = pd.read_csv(csv_url)
    df_remote.to_csv(file_path, index=False)
    print("✅ Dataset descargado y guardado en 'data/results.csv'")

df = pd.read_csv(file_path)
df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])

df['home_team_rank'] = df['home_team'].astype('category').cat.codes % 100
df['away_team_rank'] = df['away_team'].astype('category').cat.codes % 100

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'win'
    elif row['home_score'] < row['away_score']:
        return 'loss'
    else:
        return 'draw'

df['result'] = df.apply(get_result, axis=1)

le_team = LabelEncoder()
teams = pd.concat([df['home_team'], df['away_team']]).unique()
le_team.fit(teams)
df['home_team_encoded'] = le_team.transform(df['home_team'])
df['away_team_encoded'] = le_team.transform(df['away_team'])

le_result = LabelEncoder()
y = le_result.fit_transform(df['result'])

X = df[['home_team_encoded', 'away_team_encoded', 'home_team_rank', 'away_team_rank']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

joblib.dump(model, 'models/model.pkl')
joblib.dump(le_team, 'models/team_encoder.pkl')
joblib.dump(le_result, 'models/label_encoder_y.pkl')
print("✅ Modelo entrenado y guardado con dataset real.")
