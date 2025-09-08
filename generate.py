"""
Sophisticated Synthetic Dataset Generator for Skin Allergy Risk Prediction
This script generates a realistic dataset with proper medical correlations and distributions.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

class SkinAllergyDatasetGenerator:
    def __init__(self, n_samples=5000, random_state=42):
        """
        Initialize the dataset generator with realistic parameters
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define realistic distributions and correlations
        self.setup_distributions()
        
    def setup_distributions(self):
        """Setup realistic distributions for various features"""
        
        # Age distribution (more allergies in children and elderly)
        self.age_weights = {
            'child': (5, 18, 0.15),      # 5-18 years, 15% of population
            'young_adult': (18, 35, 0.35), # 18-35 years, 35% of population
            'adult': (35, 55, 0.35),     # 35-55 years, 35% of population
            'elderly': (55, 80, 0.15)    # 55-80 years, 15% of population
        }
        
        # Common allergens with realistic prevalence
        self.allergens = {
            'nickel': 0.15,           # 15% prevalence
            'fragrance': 0.12,        # 12% prevalence
            'latex': 0.08,            # 8% prevalence
            'preservatives': 0.10,    # 10% prevalence
            'dyes': 0.06,             # 6% prevalence
            'formaldehyde': 0.04,     # 4% prevalence
            'none': 0.45              # 45% no known allergens
        }
        
        # Skin types (Fitzpatrick scale)
        self.skin_types = {
            'Type_I': 0.05,    # Very fair, always burns
            'Type_II': 0.15,   # Fair, usually burns
            'Type_III': 0.25,  # Medium, sometimes burns
            'Type_IV': 0.25,   # Olive, rarely burns
            'Type_V': 0.20,    # Brown, very rarely burns
            'Type_VI': 0.10    # Dark brown/black, never burns
        }
        
        # Occupational exposure risks
        self.occupations = {
            'healthcare': {'exposure_risk': 0.8, 'prevalence': 0.12},
            'beauty_cosmetics': {'exposure_risk': 0.9, 'prevalence': 0.05},
            'cleaning_services': {'exposure_risk': 0.85, 'prevalence': 0.08},
            'manufacturing': {'exposure_risk': 0.7, 'prevalence': 0.15},
            'food_service': {'exposure_risk': 0.6, 'prevalence': 0.10},
            'office_work': {'exposure_risk': 0.2, 'prevalence': 0.30},
            'education': {'exposure_risk': 0.3, 'prevalence': 0.10},
            'other': {'exposure_risk': 0.4, 'prevalence': 0.10}
        }
        
    def generate_demographics(self):
        """Generate realistic demographic data"""
        demographics = []
        
        for _ in range(self.n_samples):
            # Age generation with realistic distribution
            age_group = np.random.choice(
                list(self.age_weights.keys()),
                p=[weight[2] for weight in self.age_weights.values()]
            )
            min_age, max_age, _ = self.age_weights[age_group]
            age = np.random.randint(min_age, max_age + 1)
            
            # Gender with slight female bias (more reported allergies)
            gender = np.random.choice(['Male', 'Female'], p=[0.45, 0.55])
            
            # Skin type based on demographic factors
            skin_type = np.random.choice(
                list(self.skin_types.keys()),
                p=list(self.skin_types.values())
            )
            
            # Occupation
            occupation = np.random.choice(
                list(self.occupations.keys()),
                p=[occ['prevalence'] for occ in self.occupations.values()]
            )
            
            demographics.append({
                'age': age,
                'gender': gender,
                'skin_type': skin_type,
                'occupation': occupation,
                'age_group': age_group
            })
            
        return demographics
    
    def generate_medical_history(self, demographics):
        """Generate medical history with realistic correlations"""
        medical_data = []
        
        for demo in demographics:
            age = demo['age']
            age_group = demo['age_group']
            
            # Family history of allergies (genetic component)
            family_history = np.random.choice([0, 1], p=[0.65, 0.35])
            
            # Previous allergic reactions (age-dependent)
            if age_group == 'child':
                prev_allergies_prob = 0.25 + (0.15 * family_history)
            elif age_group == 'elderly':
                prev_allergies_prob = 0.20 + (0.10 * family_history)
            else:
                prev_allergies_prob = 0.15 + (0.10 * family_history)
            
            prev_allergies = np.random.choice([0, 1], p=[1-prev_allergies_prob, prev_allergies_prob])
            
            # Asthma correlation with skin allergies
            asthma_prob = 0.08 + (0.15 * prev_allergies) + (0.10 * family_history)
            asthma = np.random.choice([0, 1], p=[1-asthma_prob, asthma_prob])
            
            # Eczema/Atopic dermatitis
            eczema_prob = 0.10 + (0.20 * prev_allergies) + (0.15 * family_history)
            if age_group == 'child':
                eczema_prob += 0.15  # Higher in children
            eczema = np.random.choice([0, 1], p=[1-eczema_prob, eczema_prob])
            
            # Autoimmune conditions
            autoimmune_prob = 0.05 + (0.08 * prev_allergies)
            if demo['gender'] == 'Female':
                autoimmune_prob += 0.03  # Higher in females
            autoimmune = np.random.choice([0, 1], p=[1-autoimmune_prob, autoimmune_prob])
            
            # Known allergens
            allergen_probs = list(self.allergens.values())
            if prev_allergies:
                # Increase probability of having known allergens
                allergen_probs = [p * 1.5 if k != 'none' else p * 0.3 
                                for k, p in self.allergens.items()]
                # Normalize probabilities
                total = sum(allergen_probs)
                allergen_probs = [p/total for p in allergen_probs]
            
            known_allergen = np.random.choice(
                list(self.allergens.keys()),
                p=allergen_probs
            )
            
            medical_data.append({
                'family_history_allergies': family_history,
                'previous_allergic_reactions': prev_allergies,
                'asthma': asthma,
                'eczema': eczema,
                'autoimmune_conditions': autoimmune,
                'known_primary_allergen': known_allergen
            })
            
        return medical_data
    
    def generate_environmental_factors(self):
        """Generate environmental exposure data"""
        environmental_data = []
        
        # Seasonal distribution
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
        for _ in range(self.n_samples):
            # Season (affects pollen exposure)
            season = np.random.choice(seasons)
            
            # Environmental factors based on season
            if season == 'Spring':
                pollen_count = np.random.normal(85, 20)  # High pollen
                humidity = np.random.normal(65, 15)
                temperature = np.random.normal(18, 8)
            elif season == 'Summer':
                pollen_count = np.random.normal(60, 25)
                humidity = np.random.normal(70, 20)
                temperature = np.random.normal(28, 6)
            elif season == 'Fall':
                pollen_count = np.random.normal(40, 15)
                humidity = np.random.normal(55, 18)
                temperature = np.random.normal(15, 10)
            else:  # Winter
                pollen_count = np.random.normal(20, 10)
                humidity = np.random.normal(45, 20)
                temperature = np.random.normal(5, 12)
            
            # Ensure realistic bounds
            pollen_count = max(0, min(200, pollen_count))
            humidity = max(20, min(95, humidity))
            temperature = max(-10, min(45, temperature))
            
            # Air quality index (0-500 scale)
            aqi = np.random.lognormal(3.5, 0.8)  # Log-normal distribution
            aqi = max(0, min(500, aqi))
            
            # UV index (0-12 scale)
            if season in ['Summer', 'Spring']:
                uv_index = np.random.normal(7, 2)
            else:
                uv_index = np.random.normal(3, 1.5)
            uv_index = max(0, min(12, uv_index))
            
            environmental_data.append({
                'season': season,
                'pollen_count': round(pollen_count, 1),
                'humidity_percent': round(humidity, 1),
                'temperature_celsius': round(temperature, 1),
                'air_quality_index': round(aqi, 1),
                'uv_index': round(uv_index, 1)
            })
            
        return environmental_data
    
    def generate_lifestyle_factors(self, demographics, medical_data):
        """Generate lifestyle factors with realistic correlations"""
        lifestyle_data = []
        
        for demo, medical in zip(demographics, medical_data):
            age = demo['age']
            age_group = demo['age_group']
            
            # Stress level (1-10 scale) - correlated with age and medical conditions
            base_stress = 4
            if age_group == 'young_adult' or age_group == 'adult':
                base_stress += 1.5  # Higher stress in working age
            if medical['autoimmune_conditions'] or medical['eczema']:
                base_stress += 1  # Chronic conditions increase stress
            
            stress_level = np.random.normal(base_stress, 2)
            stress_level = max(1, min(10, round(stress_level)))
            
            # Sleep quality (1-10 scale) - inversely related to stress and age
            sleep_quality = 8 - (stress_level * 0.3) + np.random.normal(0, 1)
            if age > 60:
                sleep_quality -= 1  # Sleep quality decreases with age
            sleep_quality = max(1, min(10, round(sleep_quality)))
            
            # Exercise frequency (days per week)
            if age_group == 'child':
                exercise_freq = np.random.normal(5, 1.5)  # Active children
            elif age_group == 'elderly':
                exercise_freq = np.random.normal(2.5, 1.2)  # Less active elderly
            else:
                exercise_freq = np.random.normal(3.5, 2)
            exercise_freq = max(0, min(7, round(exercise_freq)))
            
            # Diet type
            diet_probs = [0.75, 0.15, 0.08, 0.02]  # Omnivore, Vegetarian, Vegan, Other
            diet_type = np.random.choice(
                ['Omnivore', 'Vegetarian', 'Vegan', 'Other'],
                p=diet_probs
            )
            
            # Smoking status (affects skin health)
            if age < 18:
                smoking = 'Never'
            else:
                smoking_probs = [0.70, 0.15, 0.15]  # Never, Former, Current
                if age > 50:
                    smoking_probs = [0.60, 0.30, 0.10]  # More former smokers in older age
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=smoking_probs)
            
            # Alcohol consumption (drinks per week)
            if age < 21:
                alcohol_consumption = 0
            else:
                alcohol_consumption = max(0, np.random.normal(3, 4))
                alcohol_consumption = min(20, round(alcohol_consumption))
            
            lifestyle_data.append({
                'stress_level': stress_level,
                'sleep_quality_score': sleep_quality,
                'exercise_frequency_per_week': exercise_freq,
                'diet_type': diet_type,
                'smoking_status': smoking,
                'alcohol_consumption_per_week': alcohol_consumption
            })
            
        return lifestyle_data
    
    def generate_product_exposure(self, demographics):
        """Generate cosmetic and product exposure data"""
        exposure_data = []
        
        for demo in demographics:
            age = demo['age']
            gender = demo['gender']
            occupation = demo['occupation']
            
            # Cosmetic usage (higher in females and certain occupations)
            base_cosmetic_score = 2
            if gender == 'Female':
                base_cosmetic_score += 4
            if age >= 13 and age <= 35:
                base_cosmetic_score += 2  # Peak cosmetic usage age
            if occupation in ['beauty_cosmetics', 'healthcare']:
                base_cosmetic_score += 1
            
            cosmetic_usage = max(0, min(10, np.random.normal(base_cosmetic_score, 2)))
            cosmetic_usage = round(cosmetic_usage)
            
            # Skincare routine frequency
            skincare_freq = max(0, min(10, np.random.normal(cosmetic_usage * 0.8, 1.5)))
            skincare_freq = round(skincare_freq)
            
            # Hair product usage
            hair_product_usage = max(0, min(10, np.random.normal(cosmetic_usage * 0.6, 2)))
            hair_product_usage = round(hair_product_usage)
            
            # Perfume/fragrance usage
            fragrance_usage = max(0, min(10, np.random.normal(cosmetic_usage * 0.7, 2)))
            fragrance_usage = round(fragrance_usage)
            
            # Household chemical exposure
            household_chemicals = np.random.normal(5, 2)
            household_chemicals = max(1, min(10, round(household_chemicals)))
            
            # Occupational chemical exposure
            occ_exposure = self.occupations[occupation]['exposure_risk'] * 10
            occ_exposure += np.random.normal(0, 1)
            occupational_exposure = max(1, min(10, round(occ_exposure)))
            
            # Recently tried new products (binary)
            new_product_prob = 0.15 + (cosmetic_usage * 0.02)
            new_products_recently = np.random.choice([0, 1], p=[1-new_product_prob, new_product_prob])
            
            exposure_data.append({
                'cosmetic_usage_frequency': cosmetic_usage,
                'skincare_routine_frequency': skincare_freq,
                'hair_product_usage': hair_product_usage,
                'fragrance_usage_frequency': fragrance_usage,
                'household_chemical_exposure': household_chemicals,
                'occupational_chemical_exposure': occupational_exposure,
                'new_products_tried_recently': new_products_recently
            })
            
        return exposure_data
    
    def calculate_allergy_risk(self, demographics, medical_data, environmental_data, 
                             lifestyle_data, exposure_data):
        """Calculate realistic allergy risk scores based on all factors"""
        risk_scores = []
        
        for demo, med, env, life, exp in zip(demographics, medical_data, 
                                           environmental_data, lifestyle_data, exposure_data):
            
            # Base risk score
            risk_score = 10
            
            # Demographic factors
            if demo['age_group'] in ['child', 'elderly']:
                risk_score += 15  # Higher risk in children and elderly
            
            if demo['gender'] == 'Female':
                risk_score += 5  # Slightly higher reported rates in females
            
            if demo['skin_type'] in ['Type_I', 'Type_II']:
                risk_score += 10  # Fair skin more sensitive
            
            # Medical history (strongest predictors)
            if med['family_history_allergies']:
                risk_score += 20
            if med['previous_allergic_reactions']:
                risk_score += 25  # Strongest predictor
            if med['asthma']:
                risk_score += 15
            if med['eczema']:
                risk_score += 20
            if med['autoimmune_conditions']:
                risk_score += 12
            if med['known_primary_allergen'] != 'none':
                risk_score += 18
            
            # Environmental factors
            if env['season'] == 'Spring':
                risk_score += 8  # High pollen season
            if env['pollen_count'] > 80:
                risk_score += 6
            if env['humidity_percent'] > 70 or env['humidity_percent'] < 30:
                risk_score += 4  # Extreme humidity
            if env['air_quality_index'] > 100:
                risk_score += 5
            if env['uv_index'] > 8:
                risk_score += 3
            
            # Lifestyle factors
            if life['stress_level'] > 7:
                risk_score += 8
            if life['sleep_quality_score'] < 5:
                risk_score += 6
            if life['smoking_status'] == 'Current':
                risk_score += 7
            if life['diet_type'] == 'Other':  # May indicate food allergies
                risk_score += 3
            
            # Product exposure
            if exp['cosmetic_usage_frequency'] > 7:
                risk_score += 8
            if exp['new_products_tried_recently']:
                risk_score += 12  # New products increase risk
            if exp['occupational_chemical_exposure'] > 7:
                risk_score += 10
            if exp['household_chemical_exposure'] > 8:
                risk_score += 5
            
            # Add some random variation to make it more realistic
            risk_score += np.random.normal(0, 5)
            
            # Ensure score is within reasonable bounds
            risk_score = max(0, min(100, risk_score))
            
            risk_scores.append(round(risk_score, 1))
        
        return risk_scores
    
    def generate_complete_dataset(self):
        """Generate the complete synthetic dataset"""
        print("Generating demographic data...")
        demographics = self.generate_demographics()
        
        print("Generating medical history...")
        medical_data = self.generate_medical_history(demographics)
        
        print("Generating environmental factors...")
        environmental_data = self.generate_environmental_factors()
        
        print("Generating lifestyle factors...")
        lifestyle_data = self.generate_lifestyle_factors(demographics, medical_data)
        
        print("Generating product exposure data...")
        exposure_data = self.generate_product_exposure(demographics)
        
        print("Calculating allergy risk scores...")
        risk_scores = self.calculate_allergy_risk(
            demographics, medical_data, environmental_data, 
            lifestyle_data, exposure_data
        )
        
        # Combine all data
        complete_data = []
        for i in range(self.n_samples):
            row = {}
            row.update(demographics[i])
            row.update(medical_data[i])
            row.update(environmental_data[i])
            row.update(lifestyle_data[i])
            row.update(exposure_data[i])
            row['allergy_risk_score'] = risk_scores[i]
            
            # Create categorical risk levels
            if risk_scores[i] >= 70:
                row['allergy_risk_level'] = 'High'
            elif risk_scores[i] >= 40:
                row['allergy_risk_level'] = 'Medium'
            else:
                row['allergy_risk_level'] = 'Low'
            
            # Remove temporary columns
            del row['age_group']
            
            complete_data.append(row)
        
        return pd.DataFrame(complete_data)
    
    def save_dataset(self, df, filename='skin_allergy_dataset.csv'):
        """Save the dataset to CSV file with metadata"""
        
        # Save main dataset
        df.to_csv(filename, index=False)
        print(f"Dataset saved as '{filename}'")
        print(f"Dataset shape: {df.shape}")
        
        # Generate and save metadata
        metadata = {
            'dataset_info': {
                'name': 'Synthetic Skin Allergy Risk Prediction Dataset',
                'version': '1.0',
                'generated_date': datetime.now().isoformat(),
                'samples': len(df),
                'features': len(df.columns) - 2,  # Excluding target variables
                'random_seed': self.random_state
            },
            'target_variables': {
                'allergy_risk_score': 'Continuous score (0-100) indicating allergy risk',
                'allergy_risk_level': 'Categorical risk level (Low/Medium/High)'
            },
            'feature_descriptions': {
                'age': 'Age in years',
                'gender': 'Gender (Male/Female)',
                'skin_type': 'Fitzpatrick skin type (Type_I to Type_VI)',
                'occupation': 'Occupation category affecting chemical exposure',
                'family_history_allergies': 'Binary: Family history of allergies',
                'previous_allergic_reactions': 'Binary: Previous allergic reactions',
                'asthma': 'Binary: Asthma diagnosis',
                'eczema': 'Binary: Eczema/atopic dermatitis',
                'autoimmune_conditions': 'Binary: Autoimmune conditions',
                'known_primary_allergen': 'Primary known allergen or none',
                'season': 'Season of assessment',
                'pollen_count': 'Pollen count (grains per cubic meter)',
                'humidity_percent': 'Relative humidity percentage',
                'temperature_celsius': 'Temperature in Celsius',
                'air_quality_index': 'Air Quality Index (0-500)',
                'uv_index': 'UV Index (0-12)',
                'stress_level': 'Stress level (1-10 scale)',
                'sleep_quality_score': 'Sleep quality (1-10 scale)',
                'exercise_frequency_per_week': 'Exercise days per week',
                'diet_type': 'Diet type category',
                'smoking_status': 'Smoking status (Never/Former/Current)',
                'alcohol_consumption_per_week': 'Alcoholic drinks per week',
                'cosmetic_usage_frequency': 'Cosmetic usage frequency (0-10)',
                'skincare_routine_frequency': 'Skincare routine frequency (0-10)',
                'hair_product_usage': 'Hair product usage (0-10)',
                'fragrance_usage_frequency': 'Fragrance usage frequency (0-10)',
                'household_chemical_exposure': 'Household chemical exposure (1-10)',
                'occupational_chemical_exposure': 'Occupational chemical exposure (1-10)',
                'new_products_tried_recently': 'Binary: Recently tried new products'
            },
            'risk_distribution': {
                'Low': len(df[df['allergy_risk_level'] == 'Low']),
                'Medium': len(df[df['allergy_risk_level'] == 'Medium']),
                'High': len(df[df['allergy_risk_level'] == 'High'])
            }
        }
        
        # Save metadata
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved as '{metadata_filename}'")
        
        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns) - 2}")
        print("\nRisk Level Distribution:")
        print(df['allergy_risk_level'].value_counts())
        print(f"\nRisk Score Statistics:")
        print(df['allergy_risk_score'].describe())
        
        return df


def main():
    """Main function to generate the dataset"""
    print("=== Skin Allergy Risk Prediction Dataset Generator ===")
    print("This script generates a sophisticated synthetic dataset for skin allergy prediction.\n")
    
    # Get user input for dataset size
    try:
        n_samples = int(input("Enter the number of samples to generate (default 5000): ") or "5000")
    except ValueError:
        n_samples = 5000
        print("Invalid input. Using default of 5000 samples.")
    
    # Initialize generator
    generator = SkinAllergyDatasetGenerator(n_samples=n_samples)
    
    # Generate dataset
    print(f"\nGenerating {n_samples} samples...")
    df = generator.generate_complete_dataset()
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"skin_allergy_dataset_{timestamp}.csv"
    generator.save_dataset(df, filename)
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Files created:")
    print(f"  - {filename}")
    print(f"  - {filename.replace('.csv', '_metadata.json')}")
    
    # Display first few rows
    print(f"\nFirst 5 rows of the dataset:")
    print(df.head())


if __name__ == "__main__":
    main()
