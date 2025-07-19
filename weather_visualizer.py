import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class WeatherDataVisualizer:
    """
    A comprehensive weather data visualization dashboard that fetches data from OpenWeatherMap API
    and creates multiple types of visualizations using Matplotlib and Seaborn.
    """
    
    def __init__(self, api_key):
        """
        Initialize the Weather Data Visualizer
        
        Args:
            api_key (str): OpenWeatherMap API key
        """
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.weather_data = []
        
    def fetch_current_weather(self, cities):
        """
        Fetch current weather data for multiple cities
        
        Args:
            cities (list): List of city names
            
        Returns:
            pandas.DataFrame: Weather data for all cities
        """
        weather_records = []
        
        for city in cities:
            try:
                # API endpoint for current weather
                url = f"{self.base_url}/weather"
                params = {
                    'q': city,
                    'appid': self.api_key,
                    'units': 'metric'  # Celsius temperature
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract relevant weather information
                weather_record = {
                    'city': data['name'],
                    'country': data['sys']['country'],
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0),
                    'weather_main': data['weather'][0]['main'],
                    'weather_description': data['weather'][0]['description'],
                    'cloudiness': data['clouds']['all'],
                    'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
                    'sunset': datetime.fromtimestamp(data['sys']['sunset']),
                    'timezone': data['timezone'],
                    'latitude': data['coord']['lat'],
                    'longitude': data['coord']['lon'],
                    'timestamp': datetime.now()
                }
                
                weather_records.append(weather_record)
                print(f"✓ Successfully fetched data for {city}")
                
            except requests.exceptions.RequestException as e:
                print(f"✗ Error fetching data for {city}: {e}")
            except KeyError as e:
                print(f"✗ Missing data field for {city}: {e}")
                
        return pd.DataFrame(weather_records)
    
    def fetch_forecast_data(self, city, days=5):
        """
        Fetch 5-day weather forecast for a specific city
        
        Args:
            city (str): City name
            days (int): Number of days (max 5 for free API)
            
        Returns:
            pandas.DataFrame: Forecast data
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecast_records = []
            for item in data['list']:
                forecast_record = {
                    'city': data['city']['name'],
                    'datetime': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'weather_main': item['weather'][0]['main'],
                    'weather_description': item['weather'][0]['description'],
                    'cloudiness': item['clouds']['all']
                }
                forecast_records.append(forecast_record)
                
            print(f"✓ Successfully fetched forecast data for {city}")
            return pd.DataFrame(forecast_records)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching forecast for {city}: {e}")
            return pd.DataFrame()
    
    def create_comprehensive_dashboard(self, weather_df, forecast_df=None):
        """
        Create a comprehensive visualization dashboard
        
        Args:
            weather_df (pandas.DataFrame): Current weather data
            forecast_df (pandas.DataFrame): Forecast data (optional)
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('🌤️ Comprehensive Weather Data Analysis Dashboard', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Temperature comparison across cities
        plt.subplot(3, 4, 1)
        sns.barplot(data=weather_df, x='temperature', y='city', palette='coolwarm')
        plt.title('🌡️ Temperature by City', fontsize=14, fontweight='bold')
        plt.xlabel('Temperature (°C)')
        plt.grid(axis='x', alpha=0.3)
        
        # 2. Humidity vs Temperature scatter plot
        plt.subplot(3, 4, 2)
        scatter = plt.scatter(weather_df['temperature'], weather_df['humidity'], 
                            c=weather_df['pressure'], cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Pressure (hPa)')
        plt.title('💧 Humidity vs Temperature', fontsize=14, fontweight='bold')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Humidity (%)')
        plt.grid(alpha=0.3)
        
        # 3. Wind speed comparison
        plt.subplot(3, 4, 3)
        colors = plt.cm.Set3(np.linspace(0, 1, len(weather_df)))
        bars = plt.bar(weather_df['city'], weather_df['wind_speed'], color=colors)
        plt.title('💨 Wind Speed by City', fontsize=14, fontweight='bold')
        plt.ylabel('Wind Speed (m/s)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 4. Weather conditions distribution
        plt.subplot(3, 4, 4)
        weather_counts = weather_df['weather_main'].value_counts()
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(weather_counts)))
        wedges, texts, autotexts = plt.pie(weather_counts.values, labels=weather_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('☁️ Weather Conditions Distribution', fontsize=14, fontweight='bold')
        
        # 5. Pressure comparison with gradient
        plt.subplot(3, 4, 5)
        sns.barplot(data=weather_df, x='city', y='pressure', palette='plasma')
        plt.title('🌊 Atmospheric Pressure', fontsize=14, fontweight='bold')
        plt.ylabel('Pressure (hPa)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 6. Temperature vs Feels Like comparison
        plt.subplot(3, 4, 6)
        x = np.arange(len(weather_df))
        width = 0.35
        plt.bar(x - width/2, weather_df['temperature'], width, 
               label='Actual Temp', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, weather_df['feels_like'], width,
               label='Feels Like', alpha=0.8, color='orange')
        plt.title('🔥 Actual vs Feels Like Temperature', fontsize=14, fontweight='bold')
        plt.xlabel('Cities')
        plt.ylabel('Temperature (°C)')
        plt.xticks(x, weather_df['city'], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 7. Cloudiness and Visibility
        plt.subplot(3, 4, 7)
        fig2, ax1 = plt.subplots(figsize=(6, 4))
        color = 'tab:blue'
        ax1.set_xlabel('City')
        ax1.set_ylabel('Cloudiness (%)', color=color)
        ax1.bar(weather_df['city'], weather_df['cloudiness'], color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Visibility (km)', color=color)
        ax2.plot(weather_df['city'], weather_df['visibility'], 
                color=color, marker='o', linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('☁️ Cloudiness & Visibility', fontsize=14, fontweight='bold')
        plt.close(fig2)  # Close the temporary figure
        
        # Recreate plot 7 in the main subplot
        plt.subplot(3, 4, 7)
        plt.bar(weather_df['city'], weather_df['cloudiness'], alpha=0.6, label='Cloudiness (%)')
        plt.title('☁️ Cloudiness by City', fontsize=14, fontweight='bold')
        plt.ylabel('Cloudiness (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 8. Geographic distribution
        plt.subplot(3, 4, 8)
        scatter = plt.scatter(weather_df['longitude'], weather_df['latitude'],
                            c=weather_df['temperature'], s=200, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='Temperature (°C)')
        plt.title('🗺️ Geographic Temperature Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(alpha=0.3)
        
        # Add city labels
        for idx, row in weather_df.iterrows():
            plt.annotate(row['city'], (row['longitude'], row['latitude']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 9. Correlation heatmap
        plt.subplot(3, 4, 9)
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloudiness', 'visibility']
        correlation_matrix = weather_df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('🔗 Weather Parameters Correlation', fontsize=14, fontweight='bold')
        
        # 10. Forecast visualization (if available)
        if forecast_df is not None and not forecast_df.empty:
            plt.subplot(3, 4, 10)
            forecast_df['date'] = forecast_df['datetime'].dt.date
            daily_temps = forecast_df.groupby('date')['temperature'].agg(['min', 'max', 'mean'])
            
            dates = daily_temps.index
            plt.fill_between(dates, daily_temps['min'], daily_temps['max'], 
                           alpha=0.3, color='skyblue', label='Temperature Range')
            plt.plot(dates, daily_temps['mean'], marker='o', linewidth=2, 
                    color='red', label='Average Temperature')
            plt.title('📈 5-Day Temperature Forecast', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(alpha=0.3)
        
        # 11. Wind direction analysis
        plt.subplot(3, 4, 11)
        wind_directions = weather_df['wind_direction']
        theta = np.radians(wind_directions)
        colors = plt.cm.hsv(wind_directions / 360)
        
        ax = plt.subplot(3, 4, 11, projection='polar')
        bars = ax.bar(theta, weather_df['wind_speed'], color=colors, alpha=0.7)
        ax.set_title('🧭 Wind Direction & Speed', fontsize=14, fontweight='bold', pad=20)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # 12. Summary statistics table
        plt.subplot(3, 4, 12)
        plt.axis('off')
        summary_stats = weather_df[['temperature', 'humidity', 'pressure', 'wind_speed']].describe()
        table_data = []
        for stat in ['mean', 'std', 'min', 'max']:
            row = [stat.capitalize()]
            for col in summary_stats.columns:
                row.append(f"{summary_stats.loc[stat, col]:.1f}")
            table_data.append(row)
        
        table = plt.table(cellText=table_data,
                         colLabels=['Statistic'] + [col.replace('_', ' ').title() for col in summary_stats.columns],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('📊 Summary Statistics', fontsize=14, fontweight='bold', y=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()
        
        # Save the dashboard
        fig.savefig('weather_dashboard.png', dpi=300, bbox_inches='tight')
        print("📊 Dashboard saved as 'weather_dashboard.png'")
    
    def generate_insights_report(self, weather_df):
        """
        Generate detailed insights from the weather data
        
        Args:
            weather_df (pandas.DataFrame): Weather data
        """
        print("\n" + "="*60)
        print("🔍 WEATHER DATA INSIGHTS REPORT")
        print("="*60)
        
        # Temperature insights
        hottest_city = weather_df.loc[weather_df['temperature'].idxmax()]
        coldest_city = weather_df.loc[weather_df['temperature'].idxmin()]
        print(f"\n🌡️ TEMPERATURE ANALYSIS:")
        print(f"   Hottest: {hottest_city['city']} ({hottest_city['temperature']:.1f}°C)")
        print(f"   Coldest: {coldest_city['city']} ({coldest_city['temperature']:.1f}°C)")
        print(f"   Average: {weather_df['temperature'].mean():.1f}°C")
        
        # Humidity insights
        most_humid = weather_df.loc[weather_df['humidity'].idxmax()]
        least_humid = weather_df.loc[weather_df['humidity'].idxmin()]
        print(f"\n💧 HUMIDITY ANALYSIS:")
        print(f"   Most Humid: {most_humid['city']} ({most_humid['humidity']:.1f}%)")
        print(f"   Least Humid: {least_humid['city']} ({least_humid['humidity']:.1f}%)")
        print(f"   Average: {weather_df['humidity'].mean():.1f}%")
        
        # Wind insights
        windiest_city = weather_df.loc[weather_df['wind_speed'].idxmax()]
        print(f"\n💨 WIND ANALYSIS:")
        print(f"   Windiest: {windiest_city['city']} ({windiest_city['wind_speed']:.1f} m/s)")
        print(f"   Average Wind Speed: {weather_df['wind_speed'].mean():.1f} m/s")
        
        # Weather conditions
        print(f"\n☁️ WEATHER CONDITIONS:")
        for condition, count in weather_df['weather_main'].value_counts().items():
            print(f"   {condition}: {count} cities")
        
        # Pressure insights
        highest_pressure = weather_df.loc[weather_df['pressure'].idxmax()]
        lowest_pressure = weather_df.loc[weather_df['pressure'].idxmin()]
        print(f"\n🌊 PRESSURE ANALYSIS:")
        print(f"   Highest: {highest_pressure['city']} ({highest_pressure['pressure']:.1f} hPa)")
        print(f"   Lowest: {lowest_pressure['city']} ({lowest_pressure['pressure']:.1f} hPa)")
        
        print("="*60)

def main():
    """
    Main function to demonstrate the Weather Data Visualizer
    """
    # API Configuration - REPLACE WITH YOUR ACTUAL API KEY
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
    
    # Check if API key is still placeholder
    if API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        print("❌ ERROR: Please replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual API key!")
        print("Get your free API key at: https://openweathermap.org/api")
        return
    
    # Initialize the visualizer
    visualizer = WeatherDataVisualizer(API_KEY)
    
    # Define cities to analyze
    cities = [
        "London", "New York", "Tokyo", "Sydney", "Mumbai",
        "Cairo", "São Paulo", "Moscow", "Cape Town", "Singapore"
    ]
    
    print("🌍 Starting Weather Data Collection and Analysis...")
    print("="*60)
    
    try:
        # Fetch current weather data
        print("📡 Fetching current weather data...")
        weather_data = visualizer.fetch_current_weather(cities)
        
        if weather_data.empty:
            print("❌ No weather data retrieved. Please check your API key and internet connection.")
            return
        
        print(f"✅ Successfully collected data for {len(weather_data)} cities")
        
        # Fetch forecast data for one city as an example
        print("\n📈 Fetching forecast data for London...")
        forecast_data = visualizer.fetch_forecast_data("London")
        
        # Create comprehensive dashboard
        print("\n🎨 Creating comprehensive visualization dashboard...")
        visualizer.create_comprehensive_dashboard(weather_data, forecast_data)
        
        # Generate insights report
        visualizer.generate_insights_report(weather_data)
        
        # Save data to CSV for further analysis
        weather_data.to_csv('weather_data.csv', index=False)
        if not forecast_data.empty:
            forecast_data.to_csv('forecast_data.csv', index=False)
        
        print("\n✅ Analysis complete! Files saved:")
        print("   - weather_dashboard.png (Visualization dashboard)")
        print("   - weather_data.csv (Current weather data)")
        if not forecast_data.empty:
            print("   - forecast_data.csv (Forecast data)")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    print("🔑 SETUP INSTRUCTIONS:")
    print("1. Sign up at https://openweathermap.org/api")
    print("2. Get your free API key")
    print("3. Replace 'YOUR_OPENWEATHERMAP_API_KEY' in the main() function with your actual API key")
    print("4. Run the script")
    print("\n" + "="*60)
    
    # Run the main function
    main()
