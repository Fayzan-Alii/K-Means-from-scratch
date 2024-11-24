import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class IMDBMovieAnalysis:
    def __init__(self, filepath):
        """Initialize the analysis with the dataset filepath."""
        # Read CSV with specified columns
        self.df = pd.read_csv(filepath)
        self.required_columns = [
            'Title', 'Genre', 'Rating', 'Votes', 
            'Revenue (Millions)', 'Metascore', 'Year'
        ]
        self.validate_data()
        self.ratings = None
        self.genres = None
        self.cluster_labels = None
        self.centroids = None
        
    def validate_data(self):
        """Validate the data structure and handle missing values."""
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Handle missing values
        self.df['Revenue (Millions)'].fillna(0, inplace=True)
        self.df['Metascore'].fillna(self.df['Metascore'].mean(), inplace=True)
        
        # Remove any rows with missing ratings
        self.df = self.df.dropna(subset=['Rating'])
        
    def preprocess_data(self):
        """Preprocess the data and extract features."""
        # Convert rating to numerical format
        self.ratings = self.df['Rating'].values.reshape(-1, 1)
        
        # Extract unique genres (split and clean)
        all_genres = self.df['Genre'].str.split(',').explode()
        self.genres = sorted(list(set(all_genres.str.strip())))
        
        # Create genre binary matrix
        genre_matrix = pd.DataFrame()
        for genre in self.genres:
            genre_matrix[genre] = self.df['Genre'].str.contains(genre).astype(int)
            
        return self.ratings, genre_matrix
    
    def assign_rating_categories(self):
        """Assign rating categories based on the given scale."""
        conditions = [
            (self.df['Rating'] >= 8),
            (self.df['Rating'] >= 6) & (self.df['Rating'] < 8),
            (self.df['Rating'] < 6)
        ]
        choices = ['Top Rated', 'Average Rated', 'Low Rated']
        self.df['Rating_Category'] = np.select(conditions, choices)
        
        # Print summary of categories
        category_summary = self.df['Rating_Category'].value_counts()
        print("\nRating Categories Distribution:")
        print(category_summary)
        
        return self.df['Rating_Category']
    
    def kmeans_clustering(self, k=3, max_iters=100, tol=1e-4):
        """Implement K-means clustering from scratch."""
        # Initialize centroids based on rating categories
        centroids = np.array([5, 7, 8.5]).reshape(-1, 1)  # Initial guesses based on categories
        
        prev_centroids = centroids + 10*tol
        iteration = 0
        
        while iteration < max_iters and np.sum(np.abs(centroids - prev_centroids)) > tol:
            prev_centroids = centroids.copy()
            
            # Assign points to nearest centroid
            distances = np.abs(self.ratings - centroids.reshape(1, -1))
            self.cluster_labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(k):
                if np.sum(self.cluster_labels == i) > 0:
                    centroids[i] = np.mean(self.ratings[self.cluster_labels == i])
            
            iteration += 1
        
        self.centroids = centroids
        print(f"\nClustering converged after {iteration} iterations")
        return self.cluster_labels, centroids
    
    def analyze_genres_in_clusters(self):
        """Analyze genre distribution within clusters."""
        cluster_genre_analysis = pd.DataFrame()
        
        for i in range(len(self.centroids)):
            # Get movies in current cluster
            cluster_movies = self.df[self.cluster_labels == i]
            
            # Calculate genre frequencies and percentages
            genre_counts = []
            genre_percentages = []
            for genre in self.genres:
                count = cluster_movies['Genre'].str.contains(genre).sum()
                percentage = (count / len(cluster_movies)) * 100
                genre_counts.append(f"{count} ({percentage:.1f}%)")
            
            cluster_genre_analysis[f'Cluster {i} (avg rating: {self.centroids[i][0]:.2f})'] = genre_counts
        
        cluster_genre_analysis.index = self.genres
        return cluster_genre_analysis
    
    def evaluate_clustering(self):
        """Generate confusion matrix comparing cluster ratings with original ratings."""
        # Convert ratings to categories
        true_labels = pd.cut(self.df['Rating'], 
                           bins=[0, 6, 8, 10], 
                           labels=[0, 1, 2])
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_labels, self.cluster_labels)
        
        # Calculate accuracy
        accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        print(f"\nClustering Accuracy: {accuracy:.2f}")
        
        return conf_matrix
    
    def visualize_clusters(self):
        """Create visualizations for the clustering results."""
        plt.figure(figsize=(20, 6))
        
        # Plot 1: Rating Distribution with Clusters
        plt.subplot(1, 3, 1)
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        for i in range(len(self.centroids)):
            cluster_data = self.ratings[self.cluster_labels == i]
            plt.hist(cluster_data, alpha=0.5, label=f'Cluster {i} (n={len(cluster_data)})', 
                    color=colors[i])
        plt.axvline(x=6, color='r', linestyle='--', alpha=0.5, label='Rating Boundaries')
        plt.axvline(x=8, color='r', linestyle='--', alpha=0.5)
        plt.title('Rating Distribution by Cluster')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 2: Scatter plot of Ratings vs Revenue
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(self.df['Rating'], self.df['Revenue (Millions)'],
                            c=self.cluster_labels, cmap='viridis',
                            alpha=0.6)
        for i, centroid in enumerate(self.centroids):
            plt.axvline(x=centroid[0], color=colors[i], linestyle='--', alpha=0.5)
        plt.colorbar(scatter)
        plt.title('Ratings vs Revenue by Cluster')
        plt.xlabel('Rating')
        plt.ylabel('Revenue (Millions)')
        
        # Plot 3: Genre Distribution in Clusters
        genre_analysis = self.analyze_genres_in_clusters()
        plt.subplot(1, 3, 3)
        
        # Convert percentage strings to floats for visualization
        plot_data = genre_analysis.copy()
        for col in plot_data.columns:
            plot_data[col] = plot_data[col].apply(lambda x: float(x.split('(')[1].split('%')[0]))
        
        sns.heatmap(plot_data, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Genre Distribution in Clusters (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.show()

def main():
    # Initialize analysis
    analysis = IMDBMovieAnalysis('kmeans/IMDB-Movie-Data.csv')
    
    # Preprocess data
    print("Starting data preprocessing...")
    ratings, genre_matrix = analysis.preprocess_data()
    
    # Assign rating categories
    rating_categories = analysis.assign_rating_categories()
    
    # Perform clustering
    print("\nPerforming K-means clustering...")
    labels, centroids = analysis.kmeans_clustering()
    
    # Analyze genres
    print("\nAnalyzing genre distribution...")
    genre_analysis = analysis.analyze_genres_in_clusters()
    
    # Evaluate clustering
    print("\nEvaluating clustering results...")
    conf_matrix = analysis.evaluate_clustering()
    
    # Print detailed results
    print("\nCluster Centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i}: {centroid[0]:.2f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nGenre Analysis:")
    print(genre_analysis)
    
    # Visualize results
    print("\nGenerating visualizations...")
    analysis.visualize_clusters()

if __name__ == "__main__":
    main()