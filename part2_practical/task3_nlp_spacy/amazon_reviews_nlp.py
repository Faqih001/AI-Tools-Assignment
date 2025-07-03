"""
Task 3: NLP with spaCy
Text Data: Amazon Product Reviews
Goal: Perform NER to extract product names/brands, analyze sentiment

Author: [Your Team Name]
Date: [Current Date]
"""

import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import requests
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class AmazonReviewAnalyzer:
    """
    A comprehensive analyzer for Amazon product reviews using spaCy for NLP tasks.
    """
    
    def __init__(self):
        """Initialize the analyzer with spaCy model."""
        try:
            # Load English language model
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Successfully loaded spaCy English model")
        except OSError:
            print("❌ spaCy English model not found. Please install it with:")
            print("python -m spacy download en_core_web_sm")
            raise
        
        # Initialize data storage
        self.reviews = []
        self.processed_reviews = []
        self.entities = []
        self.sentiments = []
    
    def create_sample_reviews(self):
        """
        Create sample Amazon product reviews for demonstration.
        In a real scenario, you would load this from a dataset.
        """
        print("=" * 50)
        print("CREATING SAMPLE AMAZON REVIEWS DATASET")
        print("=" * 50)
        
        sample_reviews = [
            "I absolutely love my new iPhone 14 Pro from Apple! The camera quality is amazing and the battery life is fantastic. Highly recommend this product.",
            "The Samsung Galaxy S23 is decent but the price is too high. The display is beautiful but I expected better performance for the cost.",
            "This Nike Air Max shoes are incredibly comfortable. Perfect for running and daily wear. Great quality from Nike as always.",
            "Bought this Sony PlayStation 5 and it's amazing! The graphics are stunning and the games load so fast. Sony really outdid themselves.",
            "The MacBook Pro M2 from Apple is a powerhouse. Perfect for video editing and programming. The build quality is excellent as expected from Apple.",
            "This Adidas sneakers are okay but not great. The design is nice but they're not as comfortable as my previous Nike shoes.",
            "Love my new AirPods Pro! The noise cancellation is incredible. Apple continues to make amazing products.",
            "The Dell laptop I bought has been disappointing. Poor build quality and the screen is not bright enough. Would not recommend Dell products.",
            "This Kindle from Amazon is perfect for reading. The battery lasts forever and the screen is easy on the eyes. Great product from Amazon.",
            "The Google Pixel 7 camera is outstanding! The night mode is incredible. Google has really improved their smartphone camera technology.",
            "My new Tesla Model 3 is absolutely incredible! The autopilot feature is amazing and the battery range is impressive. Tesla is the future!",
            "Disappointed with this Microsoft Surface Pro. The keyboard is flimsy and the performance is not as advertised. Expected better from Microsoft.",
            "The Canon EOS R5 camera is a beast! Perfect for professional photography. The image quality is unmatched. Canon makes the best cameras.",
            "This Bose headphones are worth every penny. The sound quality is phenomenal and the noise cancellation is top-notch. Bose is the best!",
            "The LG OLED TV has stunning picture quality. The colors are vibrant and the contrast is perfect. LG makes excellent televisions.",
            "Not happy with this HP printer. It's slow and the print quality is poor. Had better experience with Canon printers.",
            "The Rolex Submariner is a masterpiece! The craftsmanship is incredible and it keeps perfect time. Rolex is truly luxury at its finest.",
            "This BMW X5 is a fantastic SUV. The driving experience is smooth and the interior is luxurious. BMW engineering is outstanding.",
            "The Nintendo Switch is perfect for gaming on the go. Great game library and the portability is excellent. Nintendo knows how to make fun consoles.",
            "Disappointed with this Fitbit tracker. The battery life is poor and the app is buggy. My previous Garmin device was much better."
        ]
        
        self.reviews = sample_reviews
        print(f"Created {len(self.reviews)} sample reviews")
        
        # Create DataFrame for better organization
        self.df = pd.DataFrame({
            'review_id': range(1, len(self.reviews) + 1),
            'review_text': self.reviews
        })
        
        print("\\nSample reviews:")
        for i, review in enumerate(self.reviews[:3], 1):
            print(f"{i}. {review}")
        print("...")
        
        return self.df
    
    def perform_ner(self):
        """
        Perform Named Entity Recognition to extract product names and brands.
        """
        print("\\n" + "=" * 50)
        print("PERFORMING NAMED ENTITY RECOGNITION")
        print("=" * 50)
        
        all_entities = []
        products = []
        brands = []
        organizations = []
        
        for i, review in enumerate(self.reviews):
            # Process the review with spaCy
            doc = self.nlp(review)
            
            review_entities = []
            for ent in doc.ents:
                entity_info = {
                    'review_id': i + 1,
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                review_entities.append(entity_info)
                all_entities.append(entity_info)
                
                # Categorize entities
                if ent.label_ in ['ORG', 'PERSON']:  # Organizations and persons (brands)
                    brands.append(ent.text)
                elif ent.label_ in ['PRODUCT', 'WORK_OF_ART']:  # Products
                    products.append(ent.text)
                elif ent.label_ == 'ORG':
                    organizations.append(ent.text)
            
            self.processed_reviews.append({
                'review_id': i + 1,
                'text': review,
                'entities': review_entities
            })
        
        self.entities = all_entities
        
        # Additional pattern-based extraction for common product patterns
        self.extract_products_with_patterns()
        
        print(f"Total entities found: {len(all_entities)}")
        print(f"\\nEntity types found:")
        entity_types = Counter([ent['label'] for ent in all_entities])
        for ent_type, count in entity_types.most_common():
            print(f"- {ent_type} ({spacy.explain(ent_type)}): {count}")
        
        return all_entities
    
    def extract_products_with_patterns(self):
        """
        Extract product names and brands using pattern matching.
        """
        print("\\n" + "=" * 30)
        print("PATTERN-BASED PRODUCT EXTRACTION")
        print("=" * 30)
        
        # Common brand patterns
        brand_patterns = [
            r'\\b(Apple|Samsung|Sony|Nike|Adidas|Google|Microsoft|Amazon|Tesla|Canon|Bose|LG|HP|Rolex|BMW|Nintendo|Fitbit|Garmin|Dell)\\b',
        ]
        
        # Common product patterns
        product_patterns = [
            r'\\b(iPhone|Galaxy|PlayStation|MacBook|AirPods|Kindle|Pixel|Model [0-9]|Surface|EOS|Submariner|Switch)\\b',
            r'\\b[A-Z][a-z]+ [A-Z][a-z]+ [0-9]+\\b',  # Product with model number
            r'\\b[A-Z][a-z]+ [A-Z][0-9]+\\b',  # Product with alphanumeric model
        ]
        
        extracted_brands = set()
        extracted_products = set()
        
        for review in self.reviews:
            # Extract brands
            for pattern in brand_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                extracted_brands.update(matches)
            
            # Extract products
            for pattern in product_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                extracted_products.update(matches)
        
        print(f"\\nExtracted Brands: {sorted(extracted_brands)}")
        print(f"Extracted Products: {sorted(extracted_products)}")
        
        # Store for analysis
        self.extracted_brands = extracted_brands
        self.extracted_products = extracted_products
    
    def analyze_sentiment_rule_based(self):
        """
        Analyze sentiment using a rule-based approach with spaCy and TextBlob.
        """
        print("\\n" + "=" * 50)
        print("RULE-BASED SENTIMENT ANALYSIS")
        print("=" * 50)
        
        # Define sentiment keywords
        positive_words = {
            'love', 'amazing', 'fantastic', 'excellent', 'great', 'perfect', 'outstanding',
            'incredible', 'wonderful', 'awesome', 'brilliant', 'superb', 'magnificent',
            'phenomenal', 'impressive', 'recommend', 'best', 'good', 'beautiful'
        }
        
        negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'bad',
            'worst', 'disappointing', 'flimsy', 'slow', 'buggy', 'expensive', 'not happy',
            'not recommend', 'not good', 'not great', 'not worth', 'waste'
        }
        
        sentiment_results = []
        
        for i, review in enumerate(self.reviews):
            # Process with spaCy for linguistic features
            doc = self.nlp(review)
            
            # Rule-based sentiment scoring
            positive_score = 0
            negative_score = 0
            
            for token in doc:
                word = token.lemma_.lower()
                if word in positive_words:
                    positive_score += 1
                elif word in negative_words:
                    negative_score += 1
            
            # Calculate sentiment using TextBlob for comparison
            blob = TextBlob(review)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Determine overall sentiment
            if positive_score > negative_score:
                rule_sentiment = 'Positive'
                rule_score = positive_score - negative_score
            elif negative_score > positive_score:
                rule_sentiment = 'Negative'
                rule_score = negative_score - positive_score
            else:
                rule_sentiment = 'Neutral'
                rule_score = 0
            
            # TextBlob sentiment
            if textblob_polarity > 0.1:
                textblob_sentiment = 'Positive'
            elif textblob_polarity < -0.1:
                textblob_sentiment = 'Negative'
            else:
                textblob_sentiment = 'Neutral'
            
            sentiment_info = {
                'review_id': i + 1,
                'text': review,
                'rule_sentiment': rule_sentiment,
                'rule_score': rule_score,
                'positive_words': positive_score,
                'negative_words': negative_score,
                'textblob_sentiment': textblob_sentiment,
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity
            }
            
            sentiment_results.append(sentiment_info)
        
        self.sentiments = sentiment_results
        
        # Summary statistics
        rule_sentiments = [s['rule_sentiment'] for s in sentiment_results]
        textblob_sentiments = [s['textblob_sentiment'] for s in sentiment_results]
        
        print(f"\\nRule-based Sentiment Distribution:")
        rule_counts = Counter(rule_sentiments)
        for sentiment, count in rule_counts.items():
            print(f"- {sentiment}: {count} ({count/len(sentiment_results)*100:.1f}%)")
        
        print(f"\\nTextBlob Sentiment Distribution:")
        textblob_counts = Counter(textblob_sentiments)
        for sentiment, count in textblob_counts.items():
            print(f"- {sentiment}: {count} ({count/len(sentiment_results)*100:.1f}%)")
        
        return sentiment_results
    
    def visualize_results(self):
        """
        Create visualizations for the analysis results.
        """
        print("\\n" + "=" * 50)
        print("VISUALIZING ANALYSIS RESULTS")
        print("=" * 50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Entity types distribution
        entity_types = [ent['label'] for ent in self.entities]
        entity_counts = Counter(entity_types)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title('Distribution of Entity Types')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 2. Sentiment distribution comparison
        plt.subplot(2, 3, 2)
        rule_sentiments = [s['rule_sentiment'] for s in self.sentiments]
        textblob_sentiments = [s['textblob_sentiment'] for s in self.sentiments]
        
        x = np.arange(3)
        width = 0.35
        
        rule_counts = Counter(rule_sentiments)
        textblob_counts = Counter(textblob_sentiments)
        
        sentiments = ['Negative', 'Neutral', 'Positive']
        rule_values = [rule_counts.get(s, 0) for s in sentiments]
        textblob_values = [textblob_counts.get(s, 0) for s in sentiments]
        
        plt.bar(x - width/2, rule_values, width, label='Rule-based', alpha=0.8)
        plt.bar(x + width/2, textblob_values, width, label='TextBlob', alpha=0.8)
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Analysis Comparison')
        plt.xticks(x, sentiments)
        plt.legend()
        
        # 3. Brand mentions
        plt.subplot(2, 3, 3)
        if hasattr(self, 'extracted_brands') and self.extracted_brands:
            brand_mentions = Counter()
            for review in self.reviews:
                for brand in self.extracted_brands:
                    if brand.lower() in review.lower():
                        brand_mentions[brand] += 1
            
            if brand_mentions:
                brands, counts = zip(*brand_mentions.most_common(10))
                plt.barh(brands, counts)
                plt.title('Top Brand Mentions')
                plt.xlabel('Mentions')
        
        # 4. Sentiment polarity distribution
        plt.subplot(2, 3, 4)
        polarities = [s['textblob_polarity'] for s in self.sentiments]
        plt.hist(polarities, bins=20, alpha=0.7, color='blue')
        plt.title('Sentiment Polarity Distribution')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 5. Subjectivity distribution
        plt.subplot(2, 3, 5)
        subjectivities = [s['textblob_subjectivity'] for s in self.sentiments]
        plt.hist(subjectivities, bins=20, alpha=0.7, color='green')
        plt.title('Subjectivity Distribution')
        plt.xlabel('Subjectivity Score')
        plt.ylabel('Frequency')
        
        # 6. Sentiment by review length
        plt.subplot(2, 3, 6)
        review_lengths = [len(review.split()) for review in self.reviews]
        polarities = [s['textblob_polarity'] for s in self.sentiments]
        
        plt.scatter(review_lengths, polarities, alpha=0.6)
        plt.title('Sentiment vs Review Length')
        plt.xlabel('Review Length (words)')
        plt.ylabel('Sentiment Polarity')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self):
        """
        Generate a detailed analysis report.
        """
        print("\\n" + "=" * 50)
        print("DETAILED ANALYSIS REPORT")
        print("=" * 50)
        
        # Entity Analysis
        print("\\n📊 ENTITY EXTRACTION RESULTS:")
        print("-" * 30)
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for ent in self.entities:
            entities_by_type[ent['label']].append(ent['text'])
        
        for ent_type, entities in entities_by_type.items():
            unique_entities = list(set(entities))
            print(f"\\n{ent_type} ({spacy.explain(ent_type)}):")
            for entity in unique_entities[:10]:  # Show top 10
                count = entities.count(entity)
                print(f"  - {entity} (mentioned {count} times)")
        
        # Brand and Product Analysis
        print("\\n🏢 BRAND AND PRODUCT ANALYSIS:")
        print("-" * 30)
        
        if hasattr(self, 'extracted_brands'):
            print(f"Unique brands identified: {len(self.extracted_brands)}")
            print(f"Brands: {', '.join(sorted(self.extracted_brands))}")
        
        if hasattr(self, 'extracted_products'):
            print(f"\\nUnique products identified: {len(self.extracted_products)}")
            print(f"Products: {', '.join(sorted(self.extracted_products))}")
        
        # Sentiment Analysis
        print("\\n😊 SENTIMENT ANALYSIS RESULTS:")
        print("-" * 30)
        
        rule_sentiments = [s['rule_sentiment'] for s in self.sentiments]
        textblob_sentiments = [s['textblob_sentiment'] for s in self.sentiments]
        
        rule_counts = Counter(rule_sentiments)
        textblob_counts = Counter(textblob_sentiments)
        
        print("Rule-based Sentiment Analysis:")
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            count = rule_counts.get(sentiment, 0)
            percentage = count / len(self.sentiments) * 100
            print(f"  - {sentiment}: {count} reviews ({percentage:.1f}%)")
        
        print("\\nTextBlob Sentiment Analysis:")
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            count = textblob_counts.get(sentiment, 0)
            percentage = count / len(self.sentiments) * 100
            print(f"  - {sentiment}: {count} reviews ({percentage:.1f}%)")
        
        # Most positive and negative reviews
        print("\\n⭐ MOST POSITIVE REVIEW:")
        most_positive = max(self.sentiments, key=lambda x: x['textblob_polarity'])
        print(f"Polarity: {most_positive['textblob_polarity']:.3f}")
        print(f"Text: {most_positive['text']}")
        
        print("\\n⭐ MOST NEGATIVE REVIEW:")
        most_negative = min(self.sentiments, key=lambda x: x['textblob_polarity'])
        print(f"Polarity: {most_negative['textblob_polarity']:.3f}")
        print(f"Text: {most_negative['text']}")
        
        # Brand sentiment analysis
        print("\\n🏢 BRAND SENTIMENT ANALYSIS:")
        print("-" * 30)
        
        if hasattr(self, 'extracted_brands'):
            brand_sentiments = defaultdict(list)
            for sentiment_data in self.sentiments:
                review_text = sentiment_data['text'].lower()
                for brand in self.extracted_brands:
                    if brand.lower() in review_text:
                        brand_sentiments[brand].append(sentiment_data['textblob_polarity'])
            
            for brand, polarities in brand_sentiments.items():
                if polarities:
                    avg_polarity = np.mean(polarities)
                    sentiment_label = 'Positive' if avg_polarity > 0.1 else 'Negative' if avg_polarity < -0.1 else 'Neutral'
                    print(f"  - {brand}: {sentiment_label} (avg polarity: {avg_polarity:.3f})")
    
    def save_results(self, output_dir='results'):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save entities to CSV
        entities_df = pd.DataFrame(self.entities)
        entities_df.to_csv(f'{output_dir}/extracted_entities.csv', index=False)
        
        # Save sentiment analysis to CSV
        sentiments_df = pd.DataFrame(self.sentiments)
        sentiments_df.to_csv(f'{output_dir}/sentiment_analysis.csv', index=False)
        
        # Save summary report
        with open(f'{output_dir}/analysis_summary.txt', 'w') as f:
            f.write("Amazon Reviews Analysis Summary\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Total reviews analyzed: {len(self.reviews)}\\n")
            f.write(f"Total entities found: {len(self.entities)}\\n")
            
            if hasattr(self, 'extracted_brands'):
                f.write(f"Unique brands: {len(self.extracted_brands)}\\n")
                f.write(f"Brands: {', '.join(sorted(self.extracted_brands))}\\n\\n")
            
            # Sentiment summary
            rule_sentiments = [s['rule_sentiment'] for s in self.sentiments]
            rule_counts = Counter(rule_sentiments)
            
            f.write("Sentiment Distribution:\\n")
            for sentiment, count in rule_counts.items():
                percentage = count / len(self.sentiments) * 100
                f.write(f"  - {sentiment}: {count} ({percentage:.1f}%)\\n")
        
        print(f"\\n✅ Results saved to '{output_dir}' directory")

def main():
    """
    Main function to run the complete NLP analysis pipeline.
    """
    print("AMAZON PRODUCT REVIEWS NLP ANALYSIS WITH spaCy")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = AmazonReviewAnalyzer()
    
    # Step 1: Create sample dataset
    df = analyzer.create_sample_reviews()
    
    # Step 2: Perform Named Entity Recognition
    entities = analyzer.perform_ner()
    
    # Step 3: Analyze sentiment
    sentiments = analyzer.analyze_sentiment_rule_based()
    
    # Step 4: Visualize results
    analyzer.visualize_results()
    
    # Step 5: Generate detailed report
    analyzer.generate_detailed_report()
    
    # Step 6: Save results
    analyzer.save_results()
    
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully analyzed {len(analyzer.reviews)} Amazon reviews")
    print(f"✅ Extracted {len(analyzer.entities)} named entities")
    print(f"✅ Performed sentiment analysis using rule-based approach")
    print(f"✅ Generated comprehensive visualizations")
    print(f"✅ Created detailed analysis report")
    print("✅ All results saved to files!")

if __name__ == "__main__":
    main()
