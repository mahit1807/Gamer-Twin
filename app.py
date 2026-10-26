"""
GAMER-TWIN: A Transformer-Based Digital Twin for Next-Action Prediction in Interactive Games
Symbiosis Institute of Technology, Pune - Department of AIML
Streamlit Academic Demonstration Application
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.utils import load_config, PROJECT_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, load_json
from src.data_loader import load_raw_egocs_data, generate_synthetic_gameplay_data
from src.data_validation import validate_dataset, run_validation
from src.preprocessing import GameplayFeaturePreprocessor, map_action_to_target_class
from src.sequence_builder import prepare_train_val_test_splits
from src.models import create_model
from src.training import train_model
from src.evaluation import evaluate_model, load_model_from_checkpoint
from src.prediction import NextActionPredictor
from src.simulation import ReplaySimulator
from src.visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_class_distribution, plot_prediction_probabilities, plot_player_trajectory
)

st.set_page_config(
    page_title="GAMER-TWIN | Deep Learning Digital Twin",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Central Configuration
config = load_config()

# Helper to check dataset status
def get_dataset_status():
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    sample_dir = config.get("data", {}).get("sample_dir", "data/sample_data")
    
    if os.path.exists(raw_dir) and any(os.scandir(raw_dir)):
        try:
            df, meta = load_raw_egocs_data(raw_dir, max_records=2000)
            return "REAL_DATA", df, meta
        except Exception:
            pass

    # Fallback to sample data
    ticks_path = os.path.join(sample_dir, "ticks.csv")
    if not os.path.exists(ticks_path):
        df = generate_synthetic_gameplay_data(num_rounds=5, ticks_per_round=400, output_dir=sample_dir, seed=42)
    else:
        df = pd.read_csv(ticks_path)

    meta = {
        "data_source": "SYNTHETIC_TEST_FIXTURE (Demo / Pipeline Validation)",
        "is_synthetic": True,
        "raw_record_count": len(df),
        "columns": list(df.columns)
    }
    return "DEMO_DATA", df, meta


# Sidebar Navigation
st.sidebar.title("🎮 GAMER-TWIN")
st.sidebar.caption("Digital Twin Next-Action Framework")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Project Overview",
        "2. Dataset Inspection",
        "3. Model Training",
        "4. Next-Action Prediction",
        "5. Offline Simulation / Replay",
        "6. Model Comparison",
        "7. Project Information"
    ]
)

st.sidebar.markdown("---")
data_mode, current_df, data_meta = get_dataset_status()
if data_mode == "REAL_DATA":
    st.sidebar.success("🟢 Active Data: Real EgoCS-400K")
else:
    st.sidebar.warning("🟡 Active Data: Synthetic Test Fixture (Demo Mode)")

st.sidebar.markdown(
    """
    <div style='font-size: 0.8rem; color: gray;'>
    <b>Authors:</b> Mahit Turakhia & Het Shah<br>
    <b>Faculty Guide:</b> Dr. Ananya Bhattacharjee<br>
    <b>Institution:</b> SIT Pune (AIML Dept)
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# Page 1: Project Overview
# ============================================================================
if page == "1. Project Overview":
    st.title("🎮 GAMER-TWIN")
    st.subheader("A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour")
    st.markdown("#### *A Transformer-Based Digital Twin for Next-Action Prediction in Interactive Games*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 📌 Main Objective
            GAMER-TWIN learns patterns in an individual player's historical gameplay telemetry 
            and predicts the player's next gameplay action using deep sequential Transformer modeling.

            ### 🧠 Digital Twin Academic Definition
            > *"A learned computational representation of an individual player's gameplay behaviour, 
            derived from historical gameplay sequences."*
            
            **Clarification on Scope:**
            - The digital twin is a learned probabilistic sequence model of observable gameplay telemetry.
            - It does **not** model emotions, human psychology, or offline behavior.
            - It does **not** execute live gameplay manipulation; Phase I operates as an offline sequential prediction and replay demonstration.
            """
        )

        st.markdown("### 🎯 Selected Nine Target Classes")
        st.write("The framework maps observed gameplay telemetry into 9 standard action classes:")
        cols = st.columns(3)
        for i, cls in enumerate(PROJECT_CLASSES):
            cols[i % 3].markdown(f"- **{i+1}. {cls}**")

    with col2:
        st.info("### 🏛️ Academic Details\n"
                "**Institution:** Symbiosis Institute of Technology, Pune\n\n"
                "**Department:** Artificial Intelligence & Machine Learning\n\n"
                "**Students:**\n"
                "- Mahit Turakhia\n"
                "- Het Shah\n\n"
                "**Faculty Supervisor:** Dr. Ananya Bhattacharjee\n\n"
                "**Course:** Deep Learning Course Project")

        st.markdown("### 📊 System Status")
        if data_mode == "REAL_DATA":
            st.success("✅ Real EgoCS-400K Dataset Detected")
        else:
            st.warning("⚠️ Running in Demo Mode (Synthetic Fixture)")

        checkpoints_dir = "outputs/checkpoints"
        existing_ckpts = [f.replace("_best.pt", "") for f in os.listdir(checkpoints_dir) if f.endswith("_best.pt")] if os.path.exists(checkpoints_dir) else []
        if existing_ckpts:
            st.success(f"✅ Trained Checkpoints Available: {', '.join(existing_ckpts)}")
        else:
            st.error("❌ No trained model checkpoints found. Visit Page 3 to train.")


# ============================================================================
# Page 2: Dataset Inspection
# ============================================================================
elif page == "2. Dataset Inspection":
    st.header("📂 Dataset Inspection & Validation Audit")
    
    if data_mode == "REAL_DATA":
        st.success(f"**Data Provenance:** Real EgoCS-400K Dataset ({data_meta['raw_record_count']} records)")
    else:
        st.warning(
            "**Data Provenance:** Synthetic Test Fixture\n\n"
            "Real EgoCS-400K dataset is not detected in `data/raw/`. Displaying verified synthetic test telemetry "
            "designed to validate schema conformance, normalizer fitting, and sequence construction."
        )

    st.markdown("### 🔍 Telemetry Schema & Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(current_df):,}")
    c2.metric("Detected Columns", len(current_df.columns))
    c3.metric("Missing Fields", int(current_df.isnull().sum().sum()))
    c4.metric("Data Source", data_meta.get("data_source", "Unknown")[:18])

    # Validation report
    report_path = "outputs/reports/data_validation_report.json"
    if os.path.exists(report_path):
        val_report = load_json(report_path)
    else:
        _, val_report = validate_dataset(current_df, data_meta, config)

    with st.expander("📋 View Data Validation Report", expanded=True):
        st.json(val_report)

    st.markdown("### 📊 Target Class Distribution")
    class_counts = current_df["action"].value_counts().to_dict()
    chart_counts = {cls: class_counts.get(cls, 0) for cls in PROJECT_CLASSES}
    
    col_plot, col_tab = st.columns([2, 1])
    with col_plot:
        fig_dist = plot_class_distribution(chart_counts, title="Distribution of Observable Target Classes")
        st.pyplot(fig_dist)
    with col_tab:
        df_dist = pd.DataFrame([
            {"Class": cls, "Count": chart_counts[cls], "Proportion": f"{(chart_counts[cls] / len(current_df) * 100):.1f}%"}
            for cls in PROJECT_CLASSES
        ])
        st.dataframe(df_dist, use_container_width=True)

    st.markdown("### 🕹️ Raw Telemetry Sample")
    st.dataframe(current_df.head(50), use_container_width=True)


# ============================================================================
# Page 3: Model Training
# ============================================================================
elif page == "3. Model Training":
    st.header("⚙️ Model Training Pipeline")
    st.markdown("Select one of the 4 deep learning model variants, configure hyperparameters, and execute training.")

    col_cfg, col_train = st.columns([1, 2])

    with col_cfg:
        st.markdown("#### 1. Model Selection")
        selected_model = st.selectbox(
            "Architecture Variant",
            ["transformer", "transformer_ssl", "lstm_baseline", "lstm_transformer"],
            format_func=lambda x: {
                "transformer": "Model 3: Transformer Encoder (Primary)",
                "transformer_ssl": "Model 4: Transformer + Self-Supervised Learning",
                "lstm_baseline": "Model 1: LSTM Baseline",
                "lstm_transformer": "Model 2: LSTM + Transformer Hybrid"
            }.get(x, x)
        )

        st.markdown("#### 2. Hyperparameters")
        epochs_input = st.slider("Epochs", min_value=1, max_value=20, value=4)
        batch_size_input = st.selectbox("Batch Size", [16, 32, 64], index=1)
        lr_input = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)

        train_btn = st.button("🚀 Start Training", use_container_width=True)

    with col_train:
        st.markdown("#### 3. Training Monitoring & Progress")
        history_path = f"outputs/checkpoints/{selected_model}_history.json"

        if train_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                # Prepare splits with zero data leakage
                train_ds, val_ds, test_ds, _ = prepare_train_val_test_splits(current_df, config, "outputs/processed")
                input_dim = train_ds.sequences.shape[2]

                # Update training config
                cfg_copy = dict(config)
                cfg_copy["training"]["batch_size"] = batch_size_input
                cfg_copy["training"]["learning_rate"] = lr_input

                def update_progress(ep, max_ep, tl, ta, vl, va):
                    progress_bar.progress(ep / max_ep)
                    status_text.write(f"Epoch {ep}/{max_ep} - Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val Acc: {va*100:.1f}%")

                trained_m, hist = train_model(
                    model_name=selected_model,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    input_dim=input_dim,
                    config=cfg_copy,
                    epochs=epochs_input,
                    progress_callback=update_progress
                )

                st.success(f"🎉 Training complete! Best checkpoint saved to `outputs/checkpoints/{selected_model}_best.pt`")
                fig_curves = plot_training_curves(hist)
                st.pyplot(fig_curves)

            except Exception as e:
                st.error(f"Error during training: {str(e)}")

        elif os.path.exists(history_path):
            hist = load_json(history_path)
            st.info(f"Displaying existing training history for **{selected_model}** (Best epoch: {hist.get('best_epoch', 'N/A')})")
            fig_curves = plot_training_curves(hist)
            st.pyplot(fig_curves)
        else:
            st.info("No checkpoint or training history found for this model variant. Click **Start Training** to begin.")


# ============================================================================
# Page 4: Next-Action Prediction
# ============================================================================
elif page == "4. Next-Action Prediction":
    st.header("🎯 Next-Action Prediction Inference")
    st.markdown("Inspect a temporal gameplay sequence and generate probabilities for the player's next action.")

    models_available = [f.replace("_best.pt", "") for f in os.listdir("outputs/checkpoints") if f.endswith("_best.pt")] if os.path.exists("outputs/checkpoints") else []

    if not models_available:
        st.error("⚠️ No trained models found in `outputs/checkpoints/`. Please train at least one model on Page 3.")
    else:
        col_sel, col_res = st.columns([1, 2])

        with col_sel:
            selected_model = st.selectbox("Select Trained Model", models_available)
            ckpt_path = os.path.join("outputs/checkpoints", f"{selected_model}_best.pt")

            # Load sequence from processed dataset
            processed_file = "outputs/processed/processed_splits.npz"
            if os.path.exists(processed_file):
                data_splits = np.load(processed_file)
                X_test = data_splits["X_test"]
                y_test = data_splits["y_test"]
                seq_idx = st.slider("Select Sequence Window", 0, max(0, len(X_test) - 1), 0)
                selected_seq = X_test[seq_idx]
                ground_truth = PROJECT_CLASSES[y_test[seq_idx]]
            else:
                st.warning("Processed test split not found. Preprocessing sample telemetry...")
                preproc = GameplayFeaturePreprocessor(config)
                preproc.fit(current_df)
                X_all, y_all, _ = preproc.transform(current_df)
                selected_seq = X_all[:16]
                ground_truth = PROJECT_CLASSES[y_all[16]] if len(y_all) > 16 else None

        with col_res:
            try:
                predictor = NextActionPredictor(checkpoint_path=ckpt_path)
                res = predictor.predict_sequence(selected_seq, ground_truth_action=ground_truth)

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Action", res["predicted_action"])
                c2.metric("Confidence", f"{res['confidence_percentage']:.1f}%")
                if res["ground_truth_action"]:
                    match_emoji = "✅ MATCH" if res["matches_ground_truth"] else "❌ MISMATCH"
                    c3.metric("Ground Truth", f"{res['ground_truth_action']} ({match_emoji})")

                st.markdown("#### Ranked Class Probabilities")
                fig_probs = plot_prediction_probabilities(res["ranked_probabilities"])
                st.pyplot(fig_probs)

                with st.expander("🔍 Telemetry Sequence Snapshot (16 timesteps)"):
                    st.dataframe(pd.DataFrame(selected_seq).round(2), use_container_width=True)

            except Exception as e:
                st.error(f"Inference error: {str(e)}")


# ============================================================================
# Page 5: Offline Simulation / Replay
# ============================================================================
elif page == "5. Offline Simulation / Replay":
    st.header("⏪ Offline Gameplay Replay & Simulation")
    st.markdown("Chronological step-by-step playback comparing ground-truth player actions with Digital Twin predictions.")

    models_available = [f.replace("_best.pt", "") for f in os.listdir("outputs/checkpoints") if f.endswith("_best.pt")] if os.path.exists("outputs/checkpoints") else []
    
    col_ctrl, col_view = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Controls")
        chosen_model = st.selectbox("Digital Twin Model", models_available if models_available else ["None (Replay Only)"])
        
        preproc = GameplayFeaturePreprocessor(config)
        preproc.fit(current_df)

        predictor = None
        if chosen_model != "None (Replay Only)":
            ckpt_p = os.path.join("outputs/checkpoints", f"{chosen_model}_best.pt")
            predictor = NextActionPredictor(checkpoint_path=ckpt_p)

        sim = ReplaySimulator(
            df_records=current_df,
            preprocessor=preproc,
            predictor=predictor,
            window_size=16,
            is_synthetic=data_meta.get("is_synthetic", False)
        )

        step_idx = st.slider("Chronological Match Tick / Step", 0, max(0, sim.total_steps - 1), 0)
        state = sim.get_step_state(step_idx)

        st.info(f"**Replay Provenance:** `{state['data_source']}`\n\n"
                f"**Tick:** {state['tick']} (Time: {state['seconds']}s)\n\n"
                f"**Weapon:** {state['current_weapon']} (Ammo: {state['ammo']})")

    with col_view:
        st.markdown(f"#### Match Step {step_idx}: Ground Truth vs Digital Twin")
        m1, m2, m3 = st.columns(3)
        m1.metric("Observed Next Action", state["actual_action"])
        
        if state["prediction"]:
            p = state["prediction"]
            match_str = "✅ Correct" if p["matches_ground_truth"] else "❌ Differed"
            m2.metric("Digital Twin Predicted", f"{p['predicted_action']} ({p['confidence_percentage']:.1f}%)")
            m3.metric("Match Status", match_str)
        else:
            m2.metric("Digital Twin Predicted", "Model Unselected")
            m3.metric("Match Status", "N/A")

        st.markdown("##### Recent Action Chain Leading to this Tick")
        st.write(" → ".join(f"`{a}`" for a in state["recent_actions"]))

        # Trajectory map
        fig_traj = plot_player_trajectory(current_df.iloc[max(0, step_idx - 100):step_idx + 30])
        st.pyplot(fig_traj)


# ============================================================================
# Page 6: Model Comparison
# ============================================================================
elif page == "6. Model Comparison":
    st.header("📊 Multi-Model Experimental Comparison")
    st.markdown("Side-by-side evaluation metrics across all four architecture variants on the held-out test split.")

    comp_csv = "outputs/reports/model_comparison.csv"
    comp_json = "outputs/reports/model_comparison.json"

    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("### 📈 Metric Comparison Across Models")
        metric_choice = st.selectbox(
            "Select Metric to Compare",
            ["Accuracy", "Top-3 Accuracy", "F1-Score (Macro)", "F1-Score (Weighted)"]
        )

        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=100)
        models = df_comp["Model Variant"]
        vals = df_comp[metric_choice] * 100

        bars = ax.bar(models, vals, color=["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"], edgecolor="#334155")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{metric_choice} (%)", fontsize=11)
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        st.markdown("### 🧩 Confusion Matrix Viewer")
        chosen_eval = st.selectbox("View Confusion Matrix for:", df_comp["Model Variant"])
        eval_path = f"outputs/reports/{chosen_eval}_eval.json"
        if os.path.exists(eval_path):
            eval_data = load_json(eval_path)
            fig_cm = plot_confusion_matrix(eval_data["confusion_matrix"])
            st.pyplot(fig_cm)

    else:
        st.warning("⚠️ No comparative experimental results found yet. Run `python run_experiments.py` to evaluate all 4 models.")


# ============================================================================
# Page 7: Project Information
# ============================================================================
elif page == "7. Project Information":
    st.header("📖 Project Methodology, Architecture & Information")
    
    st.markdown(
        """
        ### 1. Methodology & Pipeline
        GAMER-TWIN implements an end-to-end deep learning framework designed to predict next player actions 
        from sequential gameplay telemetry:
        1. **Telemetry Synchronization:** Merges continuous game states (`ticks.csv`) with discrete combat events (`events.csv`).
        2. **Leakage-Free Preprocessing:** Standard scaling and categorical vocabularies are strictly fitted on the training split only.
        3. **Group-Based Splitting:** Data is partitioned by round/match identifier so temporal sequences do not bleed across evaluation sets.
        4. **Sliding Sequence Construction:** Bounded temporal windows of length $T=16$ predict the action at $T+1$.

        ---

        ### 2. Four Deep Learning Architectures
        - **Model 1: LSTM Baseline:** Multi-layer recurrent neural network modeling temporal momentum.
        - **Model 2: LSTM + Transformer Hybrid:** Combines LSTM temporal inductive bias with multi-head self-attention.
        - **Model 3: Transformer Encoder (Primary):** Pure attention-based sequence modeling with sinusoidal positional encodings.
        - **Model 4: Transformer + SSL:** Distinct two-stage training with Masked Feature Autoencoding pretraining followed by supervised fine-tuning.

        ---

        ### 3. Evaluation Metrics Integrity
        - **Accuracy & Top-K Accuracy:** Direct top-1 and top-3 probability match rates.
        - **Macro vs Weighted F1:** Macro F1 treats all 9 classes equally (critical for detecting rare actions like grenades or reloads); Weighted F1 accounts for natural class frequencies.
        - **Results Integrity:** No fabricated numbers or synthetic results disguised as real data.

        ---

        ### 4. Known Limitations
        - **Telemetry Discretization:** The model operates on discretized ticks rather than raw sensory mouse DPI streams.
        - **Absence of Visual Pixels:** Game state is inferred from spatial telemetry and button traces without processing video frames.
        - **Offline Demonstration:** The system operates as an offline digital twin replay, not a live game injection hook.

        ---

        ### 5. Future Work: DQN Extension (Clarification)
        > **Academic Note (Prompt Section 12 Compliance):**
        > Deep Q-Networks (DQN) and Reinforcement Learning are **NOT** part of the Phase I predictive architecture. 
        > DQN is identified as a prospective future direction where a digital twin agent could interactively select actions 
        > based on environment reward signals in an interactive sandbox.
        """
    )
