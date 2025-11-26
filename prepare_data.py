# prepare_data.py
# Prepare email data for spam detection
import pandas as pd
import numpy as np

def create_sample_emails():
    """
    Create a simple dataset of emails
    In real life, we'd use thousands of emails!
    """
    # Sample email data
    # Each email has: subject, word count, exclamation marks, money words
    emails = [
        # Spam emails (label = 1)
        {"subject": "WIN MONEY NOW!!!", "word_count": 50, "exclamations": 3,
         "money_words": 5, "all_caps": 3, "is_spam": 1},
        {"subject": "Congratulations! You've won!", "word_count": 45, "exclamations": 2,
         "money_words": 4, "all_caps": 1, "is_spam": 1},
        {"subject": "FREE OFFER LIMITED TIME", "word_count": 30, "exclamations": 1,
         "money_words": 3, "all_caps": 4, "is_spam": 1},
        {"subject": "Make $$$ working from home!", "word_count": 40, "exclamations": 1,
         "money_words": 6, "all_caps": 0, "is_spam": 1},
        {"subject": "URGENT: Claim your prize", "word_count": 35, "exclamations": 0,
         "money_words": 2, "all_caps": 1, "is_spam": 1},
        {"subject": "Discount pharmacy prices!!!", "word_count": 25, "exclamations": 3,
         "money_words": 2, "all_caps": 0, "is_spam": 1},
        {"subject": "You're a WINNER!", "word_count": 20, "exclamations": 1,
         "money_words": 1, "all_caps": 1, "is_spam": 1},
        {"subject": "Get rich quick scheme", "word_count": 55, "exclamations": 0,
         "money_words": 4, "all_caps": 0, "is_spam": 1},
        # Normal emails (label = 0)
        {"subject": "Meeting tomorrow at 3pm", "word_count": 120, "exclamations": 0,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "Re: Project update", "word_count": 200, "exclamations": 0,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "Lunch plans?", "word_count": 85, "exclamations": 1,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "Your package has shipped", "word_count": 100, "exclamations": 0,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "Happy Birthday!", "word_count": 150, "exclamations": 1,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "Class cancelled today", "word_count": 75, "exclamations": 0,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "Thanks for your help", "word_count": 90, "exclamations": 0,
         "money_words": 0, "all_caps": 0, "is_spam": 0},
        {"subject": "See you tonight", "word_count": 60, "exclamations": 0,
         "money_words": 0, "all_caps": 0, "is_spam": 0}
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(emails)
    
    # Add more training samples by creating variations
    more_samples = []
    for index, email in df.iterrows():
        # Create slight variations
        for i in range(3):
            new_email = email.copy()
            # Add some random variation
            new_email['word_count'] += np.random.randint(-10, 10)
            new_email['exclamations'] += np.random.randint(-1, 2)
            new_email['exclamations'] = max(0, new_email['exclamations'])
            more_samples.append(new_email)
    
    # Add variations to dataset
    df_expanded = pd.concat([df, pd.DataFrame(more_samples)], ignore_index=True)
    return df_expanded

# Create and save the data
if __name__ == "__main__":
    data = create_sample_emails()
    data.to_csv('data/email_data.csv', index=False)
    print(f"Created dataset with {len(data)} emails")
    print(f"Spam emails: {len(data[data['is_spam']==1])}")
    print(f"Normal emails: {len(data[data['is_spam']==0])}")