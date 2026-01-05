"""
Test script for NLP models
"""
from nlp_models import DomainClassifier

print("=" * 60)
print("TESTING DOMAIN CLASSIFIER")
print("=" * 60)

# Initialize
print("\nInitializing classifier...")
classifier = DomainClassifier()
print("Classifier ready!")

# Test narrative
narrative = "Once upon a time in a village lived a young girl named Maya. She dreamed of exploring the world."
print(f"\nTesting: {narrative[:50]}...")
result = classifier.predict([narrative])
print(f"Result: {result[0]['label']} (confidence: {result[0]['confidence']:.1%})")

# Test informational
informational = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
print(f"\nTesting: {informational[:50]}...")
result = classifier.predict([informational])
print(f"Result: {result[0]['label']} (confidence: {result[0]['confidence']:.1%})")

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
