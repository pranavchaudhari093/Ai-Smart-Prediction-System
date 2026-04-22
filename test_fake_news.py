"""
Test script to verify Fake News Model predictions
Run this to check if the model is working correctly
"""

import pickle

# Load model and vectorizer
print("📂 Loading Fake News Model...")
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/fake_news_vectorizer.pkl", "rb"))

print("✅ Model loaded successfully!")

# Test cases
test_news = [
    # REAL news example
    """The government announced new climate policies today. 
    The measures include reducing carbon emissions by 50% over the next decade.
    Scientists have been studying the environmental impact for months.""",
    
    # FAKE news example
    """SHOCKING!!! Scientists admit they've been lying about climate change all along!
    You won't believe what happens next! This one weird trick will change everything!
    Click here to find out more!!!""",
    
    # Another REAL example
    """The stock market closed higher on Tuesday as investors reacted to earnings reports.
    Major technology companies saw gains, with the index rising 2.3 percent.
    Analysts attribute the increase to strong quarterly results.""",
    
    # Another FAKE example
    """BREAKING: Secret documents reveal EVERYTHING you've been lied to about!!!
    The truth will shock you! Share this before it gets deleted!
    They don't want you to know this!!!"""
]

print("\n" + "="*70)
print("TESTING FAKE NEWS MODEL")
print("="*70 + "\n")

for i, news in enumerate(test_news, 1):
    print(f"\n📰 TEST CASE {i}")
    print("-" * 70)
    
    # Transform text
    X_test = vectorizer.transform([news])
    
    # Get probabilities
    probs = model.predict_proba(X_test)[0]
    real_prob = probs[0] * 100
    fake_prob = probs[1] * 100
    
    # Get prediction
    prediction = model.predict(X_test)[0]
    
    # Display results
    print(f"Real Probability: {real_prob:.2f}%")
    print(f"Fake Probability: {fake_prob:.2f}%")
    print(f"\nPrediction: {'FAKE 🔴' if fake_prob > real_prob else 'REAL 🟢'}")
    print(f"Confidence: {max(real_prob, fake_prob):.2f}%")
    print(f"Model Output Class: {prediction} (0=Real, 1=Fake)")
    
    print("\n" + "-" * 70)

print("\n✅ Testing complete!")
print("\n💡 Expected Results:")
print("   - Test 1 & 3 should show REAL with good confidence (>60%)")
print("   - Test 2 & 4 should show FAKE with good confidence (>60%)")
print("\n⚠️  If all predictions are REAL or confidence is always low:")
print("   1. Check if the dataset has balanced classes")
print("   2. Retrain the model with better data")
print("   3. Verify the label encoding (0=REAL, 1=FAKE)")
