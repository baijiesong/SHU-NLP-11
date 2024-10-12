import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def read_email(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def process_emails(directory):
    emails = []
    labels = []
    
    for category in ['20030228_easy_ham', '20030228_easy_ham_2', '20030228_hard_ham', '20030228_spam', '20050311_spam_2']:
        label = 'ham' if 'ham' in category else 'spam'
        category_path = os.path.join(directory, category)
        file_pattern = os.path.join(category_path, '**', '[0-9]*.*')
        
        for file_path in glob.glob(file_pattern, recursive=True):
            email_content = read_email(file_path)
            emails.append(email_content)
            labels.append(label)
    
    return pd.DataFrame({'content': emails, 'label': labels})

def get_data(data_dir):
    df = process_emails(data_dir)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

if __name__ == "__main__":
    data_dir = './data'
    train_df, test_df = get_data(data_dir)
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"垃圾邮件比例: {train_df['label'].value_counts(normalize=True)['spam']:.2%}")
