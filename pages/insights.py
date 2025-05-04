import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import visualization as viz
from utils import get_icon

def show_insights():
    """Display the model insights page of the application."""
    st.title("Model Insights")
    
    # Check if detection has been run
    if not hasattr(st.session_state, 'anomalies') or st.session_state.anomalies is None:
        st.warning("Please run anomaly detection first.")
        
        if st.button("Go to Detection Page", key="goto_detection"):
            st.session_state.current_page = "detection"
            st.rerun()
        
        return
    
    # Get the model results
    model_results = st.session_state.model_results if hasattr(st.session_state, 'model_results') else {}
    
    if not model_results:
        st.error("No model results available. Please run anomaly detection first.")
        return
    
    # Show overall model comparison
    st.markdown("### Model Performance Comparison")
    viz.plot_model_comparison(model_results)
    
    # Add tabular view of model comparisons
    st.markdown("### Algorithm Performance Metrics Comparison")
    
    # Create comparison table
    comparison_data = []
    metrics_columns = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", 
                      "Training Time (sec)", "Anomalies Count", "Anomalies %"]
    
    for model_name, results in model_results.items():
        if 'metrics' in results:
            metrics = results['metrics']
            
            # Format model name for display
            display_name = model_name.replace('_', ' ').title()
            
            # Get metrics with formatting
            accuracy = f"{metrics.get('accuracy', 0):.4f}"
            precision = f"{metrics.get('precision', 0):.4f}"
            recall = f"{metrics.get('recall', 0):.4f}"
            f1_score = f"{metrics.get('f1_score', 0):.4f}"
            auc = f"{metrics.get('auc', 0):.4f}"
            
            # Get other interesting statistics
            training_time = f"{metrics.get('training_time', 0):.3f}"
            anomalies_count = len(results.get('anomalies', []))
            anomalies_percentage = f"{metrics.get('anomaly_ratio', 0)*100:.2f}%"
            
            # Add row to comparison data
            comparison_data.append([
                display_name, accuracy, precision, recall, f1_score, auc, 
                training_time, anomalies_count, anomalies_percentage
            ])
    
    # Create DataFrame and display as table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data, columns=metrics_columns)
        
        # Style the table
        st.dataframe(comparison_df, use_container_width=True, height=len(comparison_data)*60 + 40)
        
        # Add download button for the comparison table
        csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Comparison CSV",
            csv,
            "algorithm_comparison.csv",
            "text/csv",
            key="download-comparison-csv"
        )
        
        # Add insights about model comparison
        st.markdown("#### Interpretation of Model Comparison")
        
        # Determine best models for each metric
        best_models = {}
        if len(comparison_data) > 1:  # Only compare if we have multiple models
            for i, metric in enumerate(metrics_columns[1:6]):  # Only compare numerical metrics
                try:
                    # Find the best value (highest is best for all these metrics)
                    values = [float(row[i+1]) for row in comparison_data]
                    best_value = max(values)
                    best_index = values.index(best_value)
                    best_models[metric] = comparison_data[best_index][0]
                except (ValueError, IndexError):
                    continue
            
            # Display interpretation
            if best_models:
                st.markdown("Based on the comparison, here are the strengths of each algorithm:")
                for metric, model in best_models.items():
                    st.markdown(f"- **{model}** performs best in terms of **{metric}**")
                
                # Add general recommendation
                st.markdown("\n**Recommendation:** Choose the algorithm based on your specific needs:")
                st.markdown("- If minimizing false positives is crucial, prioritize **Precision**")
                st.markdown("- If detecting all anomalies is essential, focus on **Recall**")
                st.markdown("- For a balanced approach, consider **F1 Score** as it combines precision and recall")
    else:
        st.info("No comparative metrics available.")
    
    # Create tabs for each model
    model_tabs = st.tabs([model_name.replace('_', ' ').title() for model_name in model_results.keys()])
    
    for i, (model_name, results) in enumerate(model_results.items()):
        with model_tabs[i]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Show confusion matrix
                st.markdown("#### Confusion Matrix")
                if 'metrics' in results and 'confusion_matrix' in results['metrics']:
                    viz.plot_confusion_matrix(results['metrics']['confusion_matrix'], model_name.replace('_', ' ').title())
                else:
                    st.info("Confusion matrix not available.")
            
            with col2:
                # Show performance metrics
                st.markdown("#### Key Performance Metrics")
                if 'metrics' in results:
                    metrics = results['metrics']
                    
                    # Create gauge charts for key metrics
                    create_gauge_chart(metrics.get('accuracy', 0), "Accuracy", "blue", f"{model_name}_accuracy")
                    create_gauge_chart(metrics.get('precision', 0), "Precision", "green", f"{model_name}_precision")
                    create_gauge_chart(metrics.get('recall', 0), "Recall", "purple", f"{model_name}_recall")
                    create_gauge_chart(metrics.get('f1_score', 0), "F1 Score", "orange", f"{model_name}_f1_score")
                    create_gauge_chart(metrics.get('auc', 0), "AUC", "red", f"{model_name}_auc")
            
            # Show model specific visualizations
            st.markdown("#### Model-Specific Analysis")
            
            if model_name == "isolation_forest":
                show_isolation_forest_insights(results)
            elif model_name == "autoencoder":
                show_autoencoder_insights(results)
            elif model_name == "kmeans":
                show_kmeans_insights(results)
            
            # Show performance interpretation
            st.markdown("#### Performance Interpretation")
            
            if 'metrics' in results:
                metrics = results['metrics']
                accuracy = metrics.get('accuracy', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                
                performance_text = []
                
                # Interpret accuracy
                if accuracy >= 0.9:
                    performance_text.append("- **Accuracy**: Excellent overall performance, correctly identifying both normal and anomalous data points.")
                elif accuracy >= 0.8:
                    performance_text.append("- **Accuracy**: Good overall performance, with room for improvement.")
                else:
                    performance_text.append("- **Accuracy**: Performance needs improvement, consider adjusting sensitivity or using a different algorithm.")
                
                # Interpret precision
                if precision >= 0.9:
                    performance_text.append("- **Precision**: Very high precision indicates minimal false positives, with most identified anomalies being actual anomalies.")
                elif precision >= 0.7:
                    performance_text.append("- **Precision**: Good precision, with relatively few false positives.")
                else:
                    performance_text.append("- **Precision**: Low precision indicates many false positives. Consider adjusting the threshold.")
                
                # Interpret recall
                if recall >= 0.9:
                    performance_text.append("- **Recall**: Excellent ability to find most anomalies in the dataset.")
                elif recall >= 0.7:
                    performance_text.append("- **Recall**: Good recall, finding most anomalies but missing some.")
                else:
                    performance_text.append("- **Recall**: Low recall indicates many anomalies are being missed. Consider lowering the threshold.")
                
                # Display interpretation
                for text in performance_text:
                    st.markdown(text)
                
                # Model-specific recommendations
                st.markdown("#### Model Optimization Tips")
                
                if model_name == "isolation_forest":
                    st.markdown("""
                    - Increasing the number of trees can improve accuracy but increases computation time
                    - Adjust the contamination parameter based on the expected anomaly rate in your data
                    - Best for datasets with clear outliers rather than subtle pattern deviations
                    """)
                elif model_name == "autoencoder":
                    st.markdown("""
                    - Adding more layers can help detect complex patterns but may overfit
                    - Increasing epochs may improve performance if training loss continues to decrease
                    - Consider data normalization improvements if model performance is inconsistent
                    """)
                elif model_name == "kmeans":
                    st.markdown("""
                    - Experiment with different numbers of clusters for your dataset
                    - K-means works best when anomalies form distinct clusters
                    - Consider feature engineering to improve cluster separation
                    """)
    
    # Show overall comparison interpretations
    st.markdown("### Overall Interpretation")
    
    # Calculate average metrics across models
    avg_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        values = []
        for model, results in model_results.items():
            if 'metrics' in results and metric in results['metrics']:
                values.append(results['metrics'][metric])
        
        if values:
            avg_metrics[metric] = np.mean(values)
    
    # Find best model for each metric
    best_models = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        best_score = -1
        best_model = None
        
        for model, results in model_results.items():
            if 'metrics' in results and metric in results['metrics']:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = model
        
        if best_model:
            best_models[metric] = (best_model, best_score)
    
    # Create columns for overall stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Best Performing Models")
        
        for metric, (model, score) in best_models.items():
            st.markdown(f"- **Best {metric.title()}**: {model.replace('_', ' ').title()} ({score:.4f})")
    
    with col2:
        st.markdown("#### Average Performance")
        
        for metric, value in avg_metrics.items():
            st.markdown(f"- **Average {metric.title()}**: {value:.4f}")
    
    # General recommendations
    st.markdown("### General Recommendations")
    
    st.markdown("""
    Based on the model performance analysis:
    
    1. **Ensemble Approach**: Consider combining results from multiple models for more robust anomaly detection
    2. **Threshold Tuning**: Adjust anomaly thresholds based on your specific needs for precision vs. recall
    3. **Feature Importance**: Analyze which features contribute most to anomaly detection
    4. **Regular Retraining**: Update models regularly as new data patterns emerge
    5. **Validation**: Manually validate a sample of detected anomalies to ensure accuracy
    """)

def create_gauge_chart(value, title, color, unique_id=None):
    """
    Create a gauge chart for displaying metrics.
    
    Args:
        value: Metric value (0-1)
        title: Chart title
        color: Chart color
        unique_id: Optional unique identifier for the chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.5], 'color': 'lightgray'},
                {'range': [0.5, 0.7], 'color': 'gray'},
                {'range': [0.7, 0.9], 'color': 'lightgray'},
                {'range': [0.9, 1], 'color': 'darkgray'}
            ]
        }
    ))
    
    # Update layout for dark theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=200,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    chart_key = unique_id if unique_id else f"gauge_chart_{title.lower().replace(' ', '_')}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

def show_isolation_forest_insights(results):
    """
    Show insights for Isolation Forest model.
    
    Args:
        results: Dictionary with model results
    """
    if 'scores' in results:
        # Create histogram of scores
        scores = results['scores']
        anomalies = results['anomalies']
        
        # Create a mask for anomalies
        is_anomaly = np.zeros(len(scores), dtype=bool)
        is_anomaly[anomalies] = True
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Score': scores,
            'Type': ['Anomaly' if a else 'Normal' for a in is_anomaly]
        })
        
        # Create histogram
        fig = px.histogram(
            df,
            x='Score',
            color='Type',
            barmode='overlay',
            opacity=0.7,
            title='Isolation Forest Score Distribution',
            labels={'Score': 'Anomaly Score (lower = more anomalous)'}
        )
        
        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="isolation_forest_histogram")
        
        # Explanation
        st.markdown("""
        #### How to Interpret Isolation Forest Scores:
        
        - **Lower scores** indicate more anomalous points (easier to isolate)
        - **Threshold** is determined by the contamination parameter
        - **Score distribution** helps visualize the separation between normal and anomalous points
        """)

def show_autoencoder_insights(results):
    """
    Show insights for Autoencoder model.
    
    Args:
        results: Dictionary with model results
    """
    if 'reconstruction_errors' in results:
        # Get reconstruction errors and threshold
        errors = results['reconstruction_errors']
        threshold = results.get('threshold', np.percentile(errors, 95))
        anomalies = results['anomalies']
        
        # Create a mask for anomalies
        is_anomaly = np.zeros(len(errors), dtype=bool)
        is_anomaly[anomalies] = True
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Error': errors,
            'Type': ['Anomaly' if a else 'Normal' for a in is_anomaly]
        })
        
        # Create histogram
        fig = px.histogram(
            df,
            x='Error',
            color='Type',
            barmode='overlay',
            opacity=0.7,
            title='Autoencoder Reconstruction Error Distribution',
            labels={'Error': 'Reconstruction Error (higher = more anomalous)'}
        )
        
        # Add threshold line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            annotation_position="top right"
        )
        
        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="autoencoder_histogram")
        
        # Explanation
        st.markdown("""
        #### How to Interpret Autoencoder Results:
        
        - **Reconstruction error** measures how well the autoencoder can recreate the input
        - **Higher errors** indicate more anomalous points (harder to reconstruct)
        - **Threshold** separates normal from anomalous points
        - Clear separation between populations indicates good model performance
        """)

def show_kmeans_insights(results):
    """
    Show insights for K-Means model.
    
    Args:
        results: Dictionary with model results
    """
    if 'distances' in results:
        # Get distances and threshold
        distances = results['distances']
        threshold = results.get('threshold', np.percentile(distances, 95))
        anomalies = results['anomalies']
        
        # Create a mask for anomalies
        is_anomaly = np.zeros(len(distances), dtype=bool)
        is_anomaly[anomalies] = True
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Distance': distances,
            'Type': ['Anomaly' if a else 'Normal' for a in is_anomaly]
        })
        
        # Create histogram
        fig = px.histogram(
            df,
            x='Distance',
            color='Type',
            barmode='overlay',
            opacity=0.7,
            title='K-Means Distance Distribution',
            labels={'Distance': 'Distance to Cluster Center (higher = more anomalous)'}
        )
        
        # Add threshold line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            annotation_position="top right"
        )
        
        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="kmeans_histogram")
        
        # Explanation
        st.markdown("""
        #### How to Interpret K-Means Results:
        
        - **Distance to cluster center** measures how far a point is from its assigned cluster
        - **Larger distances** indicate potential anomalies (outliers from clusters)
        - **Threshold** is set based on the expected anomaly rate
        - K-Means works best when data has natural clustering tendencies
        """)
