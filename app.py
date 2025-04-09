from flask import Flask, render_template, request, jsonify
from main import chatbot, save_user_feedback, apply_rl_improvements, get_evaluation_metrics
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)
# Use LangChain's memory instead of a simple list
memory = ConversationBufferMemory(return_messages=True)

# Store context for each message to use with feedback
context_store = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json.get('message', '')
    
    if not question:
        return jsonify({'error': 'No message provided'}), 400
    
    # Call the chatbot function from main.py with memory
    response, agent_type, context = chatbot(question, memory)
    
    # Generate a unique message ID and store the context
    message_id = str(hash(question + response))[:10]
    context_store[message_id] = {
        'question': question,
        'response': response,
        'agent_type': agent_type,
        'context': context
    }
    
    return jsonify({
        'response': response,
        'agent_type': agent_type,
        'message_id': message_id
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    message_id = data.get('message_id')
    rating = data.get('rating')
    
    # New detailed metrics
    accuracy = data.get('accuracy', 0)
    coherence = data.get('coherence', 0)
    satisfaction = data.get('satisfaction', 0)
    feedback_text = data.get('feedback_text', '')
    
    if not message_id or not rating or message_id not in context_store:
        return jsonify({'error': 'Invalid feedback data'}), 400
    
    # Get stored context and save feedback with detailed metrics
    msg_data = context_store[message_id]
    save_user_feedback(
        msg_data['question'],
        msg_data['response'],
        msg_data['agent_type'],
        rating,
        msg_data['context'],
        accuracy,
        coherence,
        satisfaction,
        feedback_text
    )
    
    return jsonify({'status': 'Feedback saved successfully'})

@app.route('/reset', methods=['POST'])
def reset_conversation():
    # Add an endpoint to reset the conversation if needed
    global memory
    memory = ConversationBufferMemory(return_messages=True)
    return jsonify({'status': 'Conversation reset successfully'})

@app.route('/rl-stats', methods=['GET'])
def get_rl_stats():
    """Get statistics about the RL improvements"""
    try:
        stats = apply_rl_improvements()
        if stats:
            return jsonify(stats)
        return jsonify({'status': 'No feedback data available yet'})
    except pd.errors.ParserError:
        # Handle CSV parsing errors
        return jsonify({
            'status': 'error',
            'message': 'Error reading feedback data. Try resetting feedback file.'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get comprehensive metrics about chatbot performance"""
    try:
        metrics = get_evaluation_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/benchmarks', methods=['GET'])
def benchmark_comparisons():
    """
    Get benchmark comparisons by analyzing the average ratings per agent 
    relative to the overall average.
    """
    try:
        metrics = get_evaluation_metrics()
        benchmarks = {}
        overall_avg = metrics.get("overall", {}).get("avg_rating", 0)
        
        # Add more comprehensive benchmark data
        benchmarks["overall"] = metrics.get("overall", {})
        benchmarks["comparison"] = {}
        
        if "by_agent" in metrics:
            for agent, data in metrics["by_agent"].items():
                agent_avg = data.get("avg_rating", 0)
                benchmarks["comparison"][agent] = {
                    "avg_rating": agent_avg,
                    "difference_from_overall": agent_avg - overall_avg,
                    "percent_difference": ((agent_avg - overall_avg) / overall_avg * 100) if overall_avg else 0,
                    "accuracy_vs_overall": data.get("avg_accuracy", 0) - metrics["overall"].get("avg_accuracy", 0),
                    "coherence_vs_overall": data.get("avg_coherence", 0) - metrics["overall"].get("avg_coherence", 0),
                    "satisfaction_vs_overall": data.get("avg_satisfaction", 0) - metrics["overall"].get("avg_satisfaction", 0)
                }
        
        # Add time-based comparison if available
        if "recent" in metrics:
            benchmarks["time_comparison"] = {
                "recent": metrics["recent"],
                "rating_trend": metrics["recent"].get("avg_rating", 0) - overall_avg,
                "accuracy_trend": metrics["recent"].get("avg_accuracy", 0) - metrics["overall"].get("avg_accuracy", 0),
                "coherence_trend": metrics["recent"].get("avg_coherence", 0) - metrics["overall"].get("avg_coherence", 0),
                "satisfaction_trend": metrics["recent"].get("avg_satisfaction", 0) - metrics["overall"].get("avg_satisfaction", 0)
            }
            
        return jsonify(benchmarks)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/metrics-trends', methods=['GET'])
def metrics_trends():
    """Get trend data for metrics over time"""
    try:
        from main import feedback_store
        import pandas as pd
        
        df = pd.read_csv(feedback_store.feedback_file)
        if df.empty:
            return jsonify({"status": "No feedback data available"})
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure all required columns exist
        for col in ['rating', 'accuracy', 'coherence', 'satisfaction']:
            if col not in df.columns:
                df[col] = 0
        
        # Group by date and calculate average metrics
        df['date'] = df['timestamp'].dt.date
        trends = df.groupby('date').agg({
            'rating': 'mean',
            'accuracy': 'mean',
            'coherence': 'mean',
            'satisfaction': 'mean'
        }).reset_index()
        
        # Convert to format suitable for charts
        result = {
            "dates": [d.strftime('%Y-%m-%d') for d in trends['date']],
            "ratings": trends['rating'].tolist(),
            "accuracy": trends['accuracy'].tolist(),
            "coherence": trends['coherence'].tolist(),
            "satisfaction": trends['satisfaction'].tolist()
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in metrics-trends: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/metrics-dashboard', methods=['GET'])
def metrics_dashboard():
    """Render a dashboard for visualizing chatbot metrics"""
    return render_template('metrics_dashboard.html')

# Add a route to reset feedback data if needed
@app.route('/reset-feedback', methods=['POST'])
def reset_feedback():
    """Reset feedback data file"""
    try:
        from main import FeedbackStore
        feedback_store = FeedbackStore()
        # This will create a new file with the correct columns
        feedback_store.__init__()
        return jsonify({'status': 'Feedback data reset successfully'})
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/performance-evaluation', methods=['GET'])
def performance_evaluation():
    """Render the performance evaluation dashboard with benchmark comparisons"""
    return render_template('performance_evaluation.html')

@app.route('/detailed-benchmarks', methods=['GET'])
def detailed_benchmarks():
    """
    Get detailed benchmark comparisons with statistical significance and historical trends
    """
    try:
        from main import feedback_store
        import pandas as pd
        
        # Load feedback data
        df = pd.read_csv(feedback_store.feedback_file)
        if df.empty:
            return jsonify({"status": "No feedback data available"})
        
        # Basic metrics
        metrics = get_evaluation_metrics()
        benchmarks = {}
        overall_avg = metrics.get("overall", {}).get("avg_rating", 0)
        
        # Enhanced benchmark data with statistical significance
        benchmarks["overall"] = metrics.get("overall", {})
        benchmarks["comparison"] = {}
        
        # Add statistical significance tests
        if "by_agent" in metrics and len(df) > 5:  # Only if we have enough data
            for agent, data in metrics["by_agent"].items():
                agent_df = df[df['agent_type'] == agent]
                other_df = df[df['agent_type'] != agent]
                
                # Skip if not enough data for comparison
                if len(agent_df) < 3 or len(other_df) < 3:
                    continue
                
                # Perform t-test to check if difference is statistically significant
                t_stat, p_value = stats.ttest_ind(
                    agent_df['rating'].dropna(), 
                    other_df['rating'].dropna(),
                    equal_var=False  # Welch's t-test (doesn't assume equal variance)
                )
                
                benchmarks["comparison"][agent] = {
                    "avg_rating": data.get("avg_rating", 0),
                    "difference_from_overall": data.get("avg_rating", 0) - overall_avg,
                    "percent_difference": ((data.get("avg_rating", 0) - overall_avg) / overall_avg * 100) if overall_avg else 0,
                    "statistical_significance": {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "is_significant": p_value < 0.05
                    },
                    "sample_size": len(agent_df),
                    "confidence_interval": calculate_confidence_interval(agent_df['rating'].dropna())
                }
        
        # Add historical performance trends
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Get weekly averages for trend analysis
            df['week'] = df['timestamp'].dt.isocalendar().week
            df['year'] = df['timestamp'].dt.isocalendar().year
            
            weekly_trends = df.groupby(['year', 'week', 'agent_type']).agg({
                'rating': 'mean',
                'accuracy': 'mean',
                'coherence': 'mean',
                'satisfaction': 'mean'
            }).reset_index()
            
            # Format for frontend visualization
            benchmarks["historical_trends"] = {}
            for agent in df['agent_type'].unique():
                agent_trends = weekly_trends[weekly_trends['agent_type'] == agent]
                if not agent_trends.empty:
                    benchmarks["historical_trends"][agent] = {
                        "weeks": [f"{row['year']}-W{row['week']}" for _, row in agent_trends.iterrows()],
                        "ratings": agent_trends['rating'].tolist(),
                        "accuracy": agent_trends['accuracy'].tolist(),
                        "coherence": agent_trends['coherence'].tolist(),
                        "satisfaction": agent_trends['satisfaction'].tolist()
                    }
        
        return jsonify(benchmarks)
    except Exception as e:
        print(f"Error in detailed-benchmarks: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a data series"""
    if len(data) < 2:
        return {"lower": 0, "upper": 0, "mean": float(data.mean()) if len(data) > 0 else 0}
    
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    
    return {
        "lower": float(interval[0]),
        "upper": float(interval[1]),
        "mean": float(mean)
    }

@app.route('/agent-comparison', methods=['GET'])
def agent_comparison():
    """
    Get direct comparison between two specific agents
    """
    try:
        from main import feedback_store
        import pandas as pd
        
        agent1 = request.args.get('agent1', 'General')
        agent2 = request.args.get('agent2', 'AI')
        
        df = pd.read_csv(feedback_store.feedback_file)
        if df.empty:
            return jsonify({"status": "No feedback data available"})
        
        # Filter data for the two agents
        agent1_data = df[df['agent_type'] == agent1]
        agent2_data = df[df['agent_type'] == agent2]
        
        if agent1_data.empty or agent2_data.empty:
            return jsonify({"status": f"Not enough data for comparison between {agent1} and {agent2}"})
        
        # Calculate comparison metrics
        comparison = {
            "agents": [agent1, agent2],
            "sample_sizes": [len(agent1_data), len(agent2_data)],
            "metrics": {
                "rating": [float(agent1_data['rating'].mean()), float(agent2_data['rating'].mean())],
                "accuracy": [float(agent1_data['accuracy'].mean()), float(agent2_data['accuracy'].mean())],
                "coherence": [float(agent1_data['coherence'].mean()), float(agent2_data['coherence'].mean())],
                "satisfaction": [float(agent1_data['satisfaction'].mean()), float(agent2_data['satisfaction'].mean())]
            },
            "differences": {
                "rating": float(agent1_data['rating'].mean() - agent2_data['rating'].mean()),
                "accuracy": float(agent1_data['accuracy'].mean() - agent2_data['accuracy'].mean()),
                "coherence": float(agent1_data['coherence'].mean() - agent2_data['coherence'].mean()),
                "satisfaction": float(agent1_data['satisfaction'].mean() - agent2_data['satisfaction'].mean())
            }
        }
        
        return jsonify(comparison)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/ollama-stats', methods=['GET'])
def get_ollama_stats():
    """Get statistics about Ollama usage"""
    try:
        from main import get_ollama_statistics
        stats = get_ollama_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 