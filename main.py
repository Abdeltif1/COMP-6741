from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
import pandas as pd
from datetime import datetime
import time
import threading
from collections import defaultdict

# Initialize the LLM
model = OllamaLLM(model="llama3.2")

# Initialize Wikipedia API for knowledge integration
wikipedia = WikipediaAPIWrapper(top_k_results=3)

# Define agent system prompts
general_system_prompt = """
You are a knowledgeable assistant who excels at providing clear, accurate answers while maintaining natural conversation flow.

1. Answer {question} using:
- Your general knowledge
- Provided {context}
- Understanding from {chat_history}

2. Keep responses:
- Clear and direct
- Factual and accurate
- Natural and conversational
- Consistent with previous information

If uncertain, ask for clarification rather than making assumptions.
"""

admission_system_prompt = """
You are a Concordia University Computer Science program expert. Your knowledge covers admissions, requirements, deadlines, and program details.

1. Provide information using:
- Official Concordia CS program knowledge
- Provided {context}
- Previous discussion context from {chat_history}

2. Focus on:
- Accurate program details
- Precise requirements and deadlines
- Clear application guidance
- Consistent information

If specific details aren't available, recommend contacting Concordia's admissions office.
"""

ai_system_prompt = """
You are an AI/ML expert who explains technical concepts clearly while adapting to the user's knowledge level.

1. Explain concepts using:
- Technical AI/ML knowledge
- Provided {context}
- Understanding of user's level from {chat_history}

2. Deliver explanations that are:
- Technically accurate
- Appropriately detailed
- Clear and practical
- Built on established concepts

Use relevant examples and analogies when helpful.
"""

router_system_prompt = """
Analyze each question and route to the most appropriate specialist:

"General" - everyday topics and general knowledge
"Admission" - Concordia CS program specific questions
"AI" - artificial intelligence and machine learning topics

Consider:
- Main topic of {question}
- Context from {chat_history}
- Technical requirements
- Specific expertise needed

Respond ONLY with "General", "Admission", or "AI"
"""


# Create prompt templates for each agent
def create_agent_prompt(system_template):
    prompt_messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("""
Previous Conversation:
{chat_history}

Current Question: {question}

Context Information: {context}

Please respond to the current question while maintaining consistency with the previous conversation.
""")
    ]
    return ChatPromptTemplate.from_messages(prompt_messages)

# Create router prompt with history
router_messages = [
    SystemMessagePromptTemplate.from_template(router_system_prompt),
    HumanMessagePromptTemplate.from_template("Question: {question}\nConversation History: {chat_history}")
]
router_prompt = ChatPromptTemplate.from_messages(router_messages)
router_chain = router_prompt | model | StrOutputParser()

# Create agent chains
general_chain = create_agent_prompt(general_system_prompt) | model | StrOutputParser()
admission_chain = create_agent_prompt(admission_system_prompt) | model | StrOutputParser()
ai_chain = create_agent_prompt(ai_system_prompt) | model | StrOutputParser()

def get_context(question, agent_type):
    try:
        if agent_type == "Admission":
            concordia_query = f"Concordia University Montreal Computer Science program {question}"
            wiki_results = wikipedia.run(concordia_query)
            return wiki_results if wiki_results else "No relevant Concordia information found."
        elif agent_type == "AI":
            return wikipedia.run(question)
        else:
            return wikipedia.run(question)
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

# Create agent chains with memory
def create_agent_with_memory(system_prompt):
    prompt_messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("Question: {question}\nContext: {context}\n\n{chat_history}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    return prompt | model | StrOutputParser()

# Create a feedback storage system
class FeedbackStore:
    def __init__(self, feedback_file="feedback_data.csv"):
        self.feedback_file = feedback_file
        # Create file with headers if it doesn't exist
        if not os.path.exists(feedback_file):
            pd.DataFrame(columns=[
                'timestamp', 'question', 'response', 'agent_type', 
                'rating', 'context_used', 'accuracy', 'coherence', 
                'satisfaction', 'feedback_text'
            ]).to_csv(feedback_file, index=False)
    
    def save_feedback(self, question, response, agent_type, rating, context, 
                     accuracy=0, coherence=0, satisfaction=0, feedback_text=""):
        """Save feedback with detailed metrics to CSV file for later training"""
        feedback_data = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'agent_type': agent_type,
            'rating': rating,
            'context_used': context,
            'accuracy': accuracy,
            'coherence': coherence,
            'satisfaction': satisfaction,
            'feedback_text': feedback_text
        }])
        
        feedback_data.to_csv(self.feedback_file, mode='a', header=False, index=False)
        return True
    
    def get_training_data(self, min_rating=4):
        """Get high-rated examples for fine-tuning"""
        if not os.path.exists(self.feedback_file):
            return pd.DataFrame()
        
        df = pd.read_csv(self.feedback_file)
        # Filter for high-quality examples
        return df[df['rating'] >= min_rating]

# Initialize feedback store
feedback_store = FeedbackStore()

# Add imports for tracking Ollama usage
import time
import threading
from collections import defaultdict

# Create a class to track Ollama usage statistics
class OllamaStats:
    def __init__(self):
        self.total_requests = 0
        self.requests_by_agent = defaultdict(int)
        self.total_tokens = 0
        self.response_times = []
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_request(self, agent_type, tokens=0, response_time=0):
        with self.lock:
            self.total_requests += 1
            self.requests_by_agent[agent_type] += 1
            self.total_tokens += tokens
            self.response_times.append(response_time)
    
    def get_stats(self):
        with self.lock:
            uptime = time.time() - self.start_time
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            return {
                "total_requests": self.total_requests,
                "requests_by_agent": dict(self.requests_by_agent),
                "total_tokens": self.total_tokens,
                "avg_tokens_per_request": self.total_tokens / self.total_requests if self.total_requests else 0,
                "avg_response_time_seconds": avg_response_time,
                "uptime_seconds": uptime,
                "requests_per_minute": (self.total_requests / uptime) * 60 if uptime > 0 else 0
            }

# Initialize the stats tracker
ollama_stats = OllamaStats()

# Modify the OllamaLLM initialization to track token usage
class TrackedOllamaLLM(OllamaLLM):
    def invoke(self, prompt, **kwargs):
        start_time = time.time()
        response = super().invoke(prompt, **kwargs)
        end_time = time.time()
        
        # Estimate token count (this is approximate)
        # A better approach would be to use a tokenizer
        input_tokens = len(prompt.split()) * 1.3  # rough estimate
        output_tokens = len(response.split()) * 1.3  # rough estimate
        
        # Record stats (agent type will be updated later)
        ollama_stats.record_request(
            "unknown", 
            tokens=int(input_tokens + output_tokens),
            response_time=end_time - start_time
        )
        
        return response

# Replace the model initialization
model = TrackedOllamaLLM(model="llama3.2")

# Modify the chatbot function to update agent type in stats
def chatbot(question, memory):
    # Use the model to detect if the input is a greeting
    greeting_check_prompt = """
    Determine if the following message is a greeting (like hello, hi, hey, etc.).
    Respond with only "yes" if it's a greeting, or "no" if it's not a greeting.
    
    Message: {question}
    """
    
    is_greeting = model.invoke(greeting_check_prompt.format(question=question)).strip().lower()
    
    if is_greeting == "yes":
        return "Hello! How can I help you today?", "General", ""
    
    # Get chat history from memory
    chat_history = memory.buffer
    
    # Get routing decision with chat history context
    router_input = {
        "question": question,
        "chat_history": chat_history
    }
    agent_type = router_chain.invoke(router_input).strip()
    
    # Update the last recorded request with the correct agent type
    with ollama_stats.lock:
        if ollama_stats.requests_by_agent["unknown"] > 0:
            ollama_stats.requests_by_agent[agent_type] += 1
            ollama_stats.requests_by_agent["unknown"] -= 1
    
    # Get context from Wikipedia based on agent type
    context = get_context(question, agent_type)
    
    # Format chat history for the agent
    formatted_history = ""
    if memory.chat_memory.messages:
        for msg in memory.chat_memory.messages:
            if hasattr(msg, 'content'):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                formatted_history += f"{role}: {msg.content}\n"
    
    # Initialize RL agent if not already done
    if not hasattr(apply_rl_improvements, 'rl_agent'):
        apply_rl_improvements.rl_agent = RLAgent()
    
    # Get the base prompt based on agent type
    if agent_type == "Admission":
        base_chain = admission_chain
    elif agent_type == "AI":
        base_chain = ai_chain
    else:
        base_chain = general_chain
    
    # Modify the prompt using learned weights
    modified_prompt = apply_rl_improvements.rl_agent.get_modified_prompt(
        base_chain.prompt.template,
        agent_type,
        context,
        formatted_history
    )
    
    # Create a new chain with the modified prompt
    modified_chain = ChatPromptTemplate.from_template(modified_prompt) | model | StrOutputParser()
    
    # Invoke the modified chain
    response = modified_chain.invoke({
        "question": question,
        "context": context,
        "chat_history": formatted_history
    })
    
    # Update memory with the new exchange
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response)
    
    # Return context along with response and agent_type
    return response, agent_type, context

# Add a function to save feedback
def save_user_feedback(question, response, agent_type, rating, context, 
                      accuracy=0, coherence=0, satisfaction=0, feedback_text=""):
    return feedback_store.save_feedback(
        question, response, agent_type, rating, context,
        accuracy, coherence, satisfaction, feedback_text
    )

class AgentPolicy:
    def __init__(self, agent_type):
        self.agent_type = agent_type
        self.learning_rate = 0.01
        self.prompt_weights = {
            'context_weight': 1.0,
            'history_weight': 1.0,
            'directness_weight': 1.0
        }
        
    def update_weights(self, reward):
        """Update weights based on feedback reward"""
        # Scale reward from 1-5 to -1 to 1
        scaled_reward = (reward - 3) / 2
        
        # Update weights using simple gradient ascent
        for key in self.prompt_weights:
            self.prompt_weights[key] += self.learning_rate * scaled_reward
            # Ensure weights stay in reasonable bounds
            self.prompt_weights[key] = max(0.5, min(1.5, self.prompt_weights[key]))

class RLAgent:
    def __init__(self):
        self.policies = {
            'General': AgentPolicy('General'),
            'Admission': AgentPolicy('Admission'),
            'AI': AgentPolicy('AI')
        }
        
    def get_modified_prompt(self, base_prompt, agent_type, context, chat_history):
        """Modify prompt based on learned weights"""
        policy = self.policies[agent_type]
        weights = policy.prompt_weights
        
        modified_prompt = f"""
        [Context Weight: {weights['context_weight']:.2f}]
        Context Information: {context}

        [History Weight: {weights['history_weight']:.2f}]
        Previous Conversation: {chat_history}

        [Directness Weight: {weights['directness_weight']:.2f}]
        Instructions:
        1. Focus on direct answers with weight {weights['directness_weight']:.2f}
        2. Reference context with weight {weights['context_weight']:.2f}
        3. Maintain conversation continuity with weight {weights['history_weight']:.2f}

        {base_prompt}
        """
        return modified_prompt

    def update_policy(self, agent_type, reward):
        """Update policy weights based on feedback"""
        if agent_type in self.policies:
            self.policies[agent_type].update_weights(reward)

def apply_rl_improvements():
    """Apply reinforcement learning improvements based on feedback"""
    if not os.path.exists(feedback_store.feedback_file):
        return None
        
    df = pd.read_csv(feedback_store.feedback_file)
    if df.empty:
        return None
    
    # Initialize RL agent if not already done
    global rl_agent
    if not hasattr(apply_rl_improvements, 'rl_agent'):
        apply_rl_improvements.rl_agent = RLAgent()
    
    # Process all feedback entries for learning
    for _, row in df.iterrows():
        apply_rl_improvements.rl_agent.update_policy(
            row['agent_type'], 
            row['rating']
        )
    
    # Calculate performance metrics
    avg_rating = df['rating'].mean()
    agent_performance = df.groupby('agent_type')['rating'].mean()
    
    # Get current policy states
    policy_states = {
        agent_type: policy.prompt_weights 
        for agent_type, policy in apply_rl_improvements.rl_agent.policies.items()
    }
    
    return {
        "avg_rating": avg_rating,
        "agent_performance": agent_performance.to_dict(),
        "current_policies": policy_states
    }

# Add a new function to get comprehensive evaluation metrics
def get_evaluation_metrics():
    """Get comprehensive evaluation metrics for the chatbot"""
    if not os.path.exists(feedback_store.feedback_file):
        return {"status": "No feedback data available"}
    
    df = pd.read_csv(feedback_store.feedback_file)
    if df.empty:
        return {"status": "No feedback data available"}
    
    # Ensure all required columns exist
    for col in ['rating', 'accuracy', 'coherence', 'satisfaction']:
        if col not in df.columns:
            df[col] = 0
    
    metrics = {
        "overall": {
            "total_interactions": len(df),
            "avg_rating": float(df['rating'].mean()),
            "avg_accuracy": float(df['accuracy'].mean()),
            "avg_coherence": float(df['coherence'].mean()),
            "avg_satisfaction": float(df['satisfaction'].mean()),
        },
        "by_agent": {}
    }
    
    # Calculate metrics by agent type
    for agent in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent]
        metrics["by_agent"][agent] = {
            "total_interactions": len(agent_df),
            "avg_rating": float(agent_df['rating'].mean()),
            "avg_accuracy": float(agent_df['accuracy'].mean()),
            "avg_coherence": float(agent_df['coherence'].mean()),
            "avg_satisfaction": float(agent_df['satisfaction'].mean()),
        }
    
    # Add time-based analysis (last 7 days vs. all time)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    recent_df = df[df['timestamp'] > (datetime.now() - pd.Timedelta(days=7))]
    
    if not recent_df.empty:
        metrics["recent"] = {
            "total_interactions": len(recent_df),
            "avg_rating": float(recent_df['rating'].mean()),
            "avg_accuracy": float(recent_df['accuracy'].mean()),
            "avg_coherence": float(recent_df['coherence'].mean()),
            "avg_satisfaction": float(recent_df['satisfaction'].mean()),
        }
    
    return metrics

# Add a function to get Ollama stats
def get_ollama_statistics():
    """Get statistics about Ollama usage"""
    stats = ollama_stats.get_stats()
    
    # Add some derived metrics
    if stats["uptime_seconds"] > 0:
        stats["tokens_per_second"] = stats["total_tokens"] / stats["uptime_seconds"]
    else:
        stats["tokens_per_second"] = 0
        
    return stats

# Main loop
def main():
    memory = ConversationBufferMemory(return_messages=True)
    print("Welcome to the Multi-Agent Chatbot!")
    print("Type 'exit' to quit, 'stats' for feedback stats, or 'ollama-stats' for Ollama usage stats.")
    
    while True:
        question = input("\nEnter a question: ")
        if question.lower() == 'exit':
            break
        elif question.lower() == 'stats':
            stats = get_evaluation_metrics()
            print("\n=== Chatbot Feedback Statistics ===")
            print(json.dumps(stats, indent=2))
            continue
        elif question.lower() == 'ollama-stats':
            stats = get_ollama_statistics()
            print("\n=== Ollama Usage Statistics ===")
            print(json.dumps(stats, indent=2))
            continue
        
        response, agent_type, context = chatbot(question, memory)
        print(f"\n[{agent_type} Agent]: {response}")
        
        # Simple evaluation metrics
        if len(memory.chat_memory.messages) > 2:
            print("\nWould you rate this response as helpful? (1-5, 5 being most helpful)")
            try:
                rating = int(input("Rating: "))
                save_user_feedback(question, response, agent_type, rating, context)
            except ValueError:
                pass

if __name__ == "__main__":
    main()





