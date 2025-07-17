# Streamlit Dashboard Code
print("\n" + "="*50)
print("📱 STREAMLIT DASHBOARD")
print("="*50)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🚌 Transport Cost Optimizer", 
    page_icon="🚌",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('transport_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ transport_data.csv not found! Please run the main script first.")
        st.stop()

# Generate sample data if file doesn't exist
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_records = 500
    
    route_ids = [f"R{i:03d}" for i in range(1, 21)]
    time_slots = ['Morning', 'Afternoon', 'Evening', 'Night']
    
    data = []
    for _ in range(n_records):
        route_id = np.random.choice(route_ids)
        time_slot = np.random.choice(time_slots)
        
        base_distance = hash(route_id) % 30 + 10
        distance_km = base_distance + np.random.normal(0, 3)
        distance_km = max(5, min(50, distance_km))
        
        fuel_multiplier = {'Morning': 1.2, 'Afternoon': 1.0, 'Evening': 1.1, 'Night': 0.9}
        fuel_cost = (2.5 + np.random.normal(0, 0.3)) * fuel_multiplier[time_slot]
        
        maintenance_cost = 0.5 * distance_km + np.random.normal(0, 2)
        maintenance_cost = max(0, maintenance_cost)
        
        passenger_multiplier = {'Morning': 0.8, 'Afternoon': 0.6, 'Evening': 0.9, 'Night': 0.3}
        avg_passengers = 25 * passenger_multiplier[time_slot]
        no_passengers = int(max(1, min(50, np.random.poisson(avg_passengers))))
        
        time_multiplier = {'Morning': 1.3, 'Afternoon': 1.0, 'Evening': 1.2, 'Night': 0.8}
        travel_duration_min = (distance_km * 2.5 + np.random.normal(0, 5)) * time_multiplier[time_slot]
        travel_duration_min = max(10, travel_duration_min)
        
        total_cost = (
            distance_km * fuel_cost +
            maintenance_cost +
            no_passengers * 0.1 +
            travel_duration_min * 0.2
        )
        
        data.append({
            'route_id': route_id,
            'distance_km': round(distance_km, 2),
            'fuel_cost': round(fuel_cost, 2),
            'maintenance_cost': round(maintenance_cost, 2),
            'no_passengers': no_passengers,
            'time_slot': time_slot,
            'travel_duration_min': round(travel_duration_min, 2),
            'total_cost': round(total_cost, 2)
        })
    
    return pd.DataFrame(data)

# Load model function
@st.cache_resource
def load_model():
    try:
        df = load_data()
    except:
        df = generate_sample_data()
    
    df_processed = df.copy()
    
    # Encode categorical features
    le_route = LabelEncoder()
    le_time = LabelEncoder()
    
    df_processed['route_id_encoded'] = le_route.fit_transform(df_processed['route_id'])
    df_processed['time_slot_encoded'] = le_time.fit_transform(df_processed['time_slot'])
    
    feature_columns = [
        'route_id_encoded', 'distance_km', 'fuel_cost', 'maintenance_cost',
        'no_passengers', 'travel_duration_min', 'time_slot_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['total_cost']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    return model, le_route, le_time, df

# Main App
def main():
    # Title and header
    st.title("🚌 Public Transport Cost Optimization Dashboard")
    st.markdown("**Analyze and predict transport costs with machine learning**")
    st.markdown("---")
    
    # Load data and model
    try:
        model, le_route, le_time, df = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Sidebar for predictions
    st.sidebar.header("🎯 Cost Prediction Tool")
    st.sidebar.markdown("Adjust parameters to predict transport costs:")
    
    # Input widgets
    route_options = sorted(df['route_id'].unique())
    selected_route = st.sidebar.selectbox("🛣️ Route ID", route_options)
    
    # Get route encoded value
    try:
        route_encoded = le_route.transform([selected_route])[0]
    except:
        route_encoded = 0
    
    distance = st.sidebar.slider("📏 Distance (km)", 5.0, 50.0, 25.0, 0.1)
    fuel_cost = st.sidebar.slider("⛽ Fuel Cost per km ($)", 1.0, 5.0, 2.5, 0.1)
    maintenance = st.sidebar.slider("🔧 Maintenance Cost ($)", 0.0, 20.0, 10.0, 0.1)
    passengers = st.sidebar.slider("👥 Number of Passengers", 1, 50, 25)
    duration = st.sidebar.slider("⏱️ Travel Duration (min)", 10.0, 150.0, 60.0, 1.0)
    
    time_options = ['Morning', 'Afternoon', 'Evening', 'Night']
    selected_time = st.sidebar.selectbox("🕐 Time Slot", time_options)
    
    try:
        time_encoded = le_time.transform([selected_time])[0]
    except:
        time_encoded = 0
    
    # Predict button
    if st.sidebar.button("🔮 Predict Cost", type="primary"):
        try:
            prediction_input = np.array([[route_encoded, distance, fuel_cost, maintenance, passengers, duration, time_encoded]])
            predicted_cost = model.predict(prediction_input)[0]
            
            st.sidebar.success(f"💰 **Predicted Cost: ${predicted_cost:.2f}**")
            
            # Show cost breakdown
            st.sidebar.info(f"""
            **Cost Breakdown:**
            - Fuel: ${distance * fuel_cost:.2f}
            - Maintenance: ${maintenance:.2f}
            - Passenger costs: ${passengers * 0.1:.2f}
            - Duration costs: ${duration * 0.2:.2f}
            """)
        except Exception as e:
            st.sidebar.error(f"Prediction error: {e}")
    
    # Main dashboard content
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_cost = df['total_cost'].mean()
        st.metric("💰 Average Cost", f"${avg_cost:.2f}")
    
    with col2:
        avg_distance = df['distance_km'].mean()
        st.metric("📏 Average Distance", f"{avg_distance:.1f} km")
    
    with col3:
        avg_passengers = df['no_passengers'].mean()
        st.metric("👥 Average Passengers", f"{avg_passengers:.0f}")
    
    with col4:
        total_routes = df['route_id'].nunique()
        st.metric("🛣️ Total Routes", f"{total_routes}")
    
    st.markdown("---")
    
    # Main content in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Cost Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['total_cost'], bins=25, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Total Cost ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Transport Costs')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("⏰ Cost by Time Slot")
        time_costs = df.groupby('time_slot')['total_cost'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(time_costs.index, time_costs.values, color=colors)
        ax.set_ylabel('Average Cost ($)')
        ax.set_title('Average Cost by Time Slot')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.1f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        st.subheader("🛣️ Top 10 Most Efficient Routes")
        route_efficiency = df.groupby('route_id').agg({
            'total_cost': 'mean',
            'distance_km': 'mean',
            'no_passengers': 'mean'
        }).round(2)
        route_efficiency['cost_per_km'] = (route_efficiency['total_cost'] / route_efficiency['distance_km']).round(2)
        
        top_routes = route_efficiency.sort_values('cost_per_km').head(10)
        st.dataframe(top_routes, use_container_width=True)
    
    st.markdown("---")
    
    # Additional analysis
    st.subheader("🔍 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Correlation Analysis", "📈 Cost Trends", "💡 Optimization Tips"])
    
    with tab1:
        st.write("**Feature Correlation Matrix**")
        numeric_cols = ['distance_km', 'fuel_cost', 'maintenance_cost', 'no_passengers', 'travel_duration_min', 'total_cost']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
    
    with tab2:
        st.write("**Distance vs Cost Analysis**")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['distance_km'], df['total_cost'], 
                           c=df['no_passengers'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Distance vs Total Cost (colored by passengers)')
        plt.colorbar(scatter, ax=ax, label='Number of Passengers')
        st.pyplot(fig)
    
    with tab3:
        st.write("**🎯 Cost Optimization Recommendations:**")
        
        # Calculate insights
        best_time = df.groupby('time_slot')['total_cost'].mean().idxmin()
        worst_time = df.groupby('time_slot')['total_cost'].mean().idxmax()
        
        st.success(f"✅ **Best Time Slot:** {best_time} (lowest average cost)")
        st.error(f"❌ **Most Expensive:** {worst_time} (highest average cost)")
        
        st.info("""
        **Key Optimization Strategies:**
        1. 🌙 **Schedule more trips during Night hours** - lowest operational costs
        2. ⛽ **Monitor fuel prices closely** - high correlation with total cost
        3. 🚌 **Optimize passenger capacity** - better cost per passenger ratio
        4. 🔧 **Implement predictive maintenance** - reduce unexpected costs
        5. 📍 **Consider route consolidation** - shorter routes are more cost-effective
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard built with Streamlit • Public Transport Cost Optimization Project*")

if __name__ == "__main__":
    main()