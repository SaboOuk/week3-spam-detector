import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

class SpamDetector:
    """Machine Learning model to detect spam emails"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None
        self.trained = False
        
    def load_data(self, filepath):
        """Load email data from CSV"""
        self.data = pd.read_csv(filepath)
        print(f"✅ Loaded {len(self.data)} emails")
        
    def prepare_features(self):
        """Prepare features for training"""
        # Features: word_count, exclamations, money_words, all_caps
        self.X = self.data[['word_count', 'exclamations', 'money_words', 'all_caps']].values
        self.y = self.data['is_spam'].values
        
        # Scale features
        self.X = self.scaler.fit_transform(self.X)
        print(f"✅ Features prepared and scaled")
        
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        print(f"✅ Data split: {len(self.X_train)} training, {len(self.X_test)} testing")
        
    def train(self):
        """Train the spam detector model"""
        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print(f"✅ Model trained successfully")
        
    def test(self):
        """Test model performance"""
        if not self.trained:
            print("❌ Model not trained yet!")
            return
            
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        
        print("\n" + "="*50)
        print("📊 MODEL PERFORMANCE")
        print("="*50)
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print("="*50)
        
    def predict_email(self, word_count, exclamations, money_words, all_caps):
        """Predict if an email is spam"""
        if not self.trained:
            print("❌ Model not trained yet!")
            return
            
        # Create feature vector and scale
        features = np.array([[word_count, exclamations, money_words, all_caps]])
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        spam_prob = probability[1] if prediction == 1 else probability[0]
        
        if prediction == 1:
            print(f"\n🚨 SPAM DETECTED! (Confidence: {spam_prob:.1%})")
        else:
            print(f"\n✅ NOT SPAM (Confidence: {spam_prob:.1%})")
            
    def visualize_data(self):
        """Create visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Spam vs Normal distribution
        spam_data = self.data[self.data['is_spam'] == 1]
        normal_data = self.data[self.data['is_spam'] == 0]
        
        # Word count
        axes[0, 0].hist([spam_data['word_count'], normal_data['word_count']], 
                        label=['Spam', 'Normal'], bins=10)
        axes[0, 0].set_title('Word Count Distribution')
        axes[0, 0].set_xlabel('Word Count')
        axes[0, 0].legend()
        
        # Exclamations
        axes[0, 1].hist([spam_data['exclamations'], normal_data['exclamations']], 
                        label=['Spam', 'Normal'], bins=5)
        axes[0, 1].set_title('Exclamation Marks Distribution')
        axes[0, 1].set_xlabel('Exclamation Marks')
        axes[0, 1].legend()
        
        # Money words
        axes[1, 0].hist([spam_data['money_words'], normal_data['money_words']], 
                        label=['Spam', 'Normal'], bins=8)
        axes[1, 0].set_title('Money Words Distribution')
        axes[1, 0].set_xlabel('Money Words Count')
        axes[1, 0].legend()
        
        # All caps
        axes[1, 1].hist([spam_data['all_caps'], normal_data['all_caps']], 
                        label=['Spam', 'Normal'], bins=6)
        axes[1, 1].set_title('ALL CAPS Words Distribution')
        axes[1, 1].set_xlabel('ALL CAPS Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('data/email_analysis.png', dpi=100)
        print("✅ Visualization saved to 'data/email_analysis.png'")
        plt.close()


def main():
    """Main program with interactive menu"""
    print("="*50)
    print("🚫 SPAM EMAIL DETECTOR")
    print("Machine Learning in Action!")
    print("="*50)
    
    detector = SpamDetector()
    
    # First, create the data if it doesn't exist
    if not os.path.exists('data/email_data.csv'):
        print("\n📝 Creating email dataset...")
        os.makedirs('data', exist_ok=True)
        from prepare_data import create_sample_emails
        data = create_sample_emails()
        data.to_csv('data/email_data.csv', index=False)
    
    # Load and prepare data
    detector.load_data('data/email_data.csv')
    detector.prepare_features()
    detector.split_data()
    
    # Train the model
    detector.train()
    detector.test()
    
    # Visualize
    print("\n📊 Creating visualizations...")
    detector.visualize_data()
    
    # Interactive testing
    while True:
        print("\n" + "="*50)
        print("TEST THE SPAM DETECTOR")
        print("="*50)
        print("1. Test a custom email")
        print("2. Test example spam email")
        print("3. Test example normal email")
        print("4. Show model performance")
        print("5. Learn about overfitting")
        print("6. Exit")
        
        choice = input("\nChoice (1-6): ")
        
        if choice == '1':
            print("\nDescribe your email:")
            try:
                words = int(input("Approximate word count: "))
                exclaim = int(input("Number of exclamation marks: "))
                money = int(input("Money-related words (free, cash, win, etc.): "))
                caps = int(input("Words in ALL CAPS: "))
                detector.predict_email(words, exclaim, money, caps)
            except ValueError:
                print("Please enter numbers only!")
                
        elif choice == '2':
            print("\nTesting spam email: 'WIN FREE CASH NOW!!!'")
            detector.predict_email(30, 3, 4, 3)
            
        elif choice == '3':
            print("\nTesting normal email: 'Meeting tomorrow at 2pm'")
            detector.predict_email(100, 0, 0, 0)
            
        elif choice == '4':
            detector.test()
            
        elif choice == '5':
            print("\n📚 ABOUT OVERFITTING")
            print("-" * 40)
            print("Overfitting is when your model:")
            print("• Memorizes the training data perfectly")
            print("• But fails on new, unseen data")
            print("\nIt's like a student who memorizes answers")
            print("but doesn't understand the concepts!")
            print("\nWe prevent it by:")
            print("• Using separate test data")
            print("• Keeping our model simple")
            print("• Having enough diverse training examples")
            
        elif choice == '6':
            print("\nThanks for using Spam Detector!")
            break


if __name__ == "__main__":
    main()
