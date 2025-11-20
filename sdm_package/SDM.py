# Force PyTorch backend BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'  # Disable TensorFlow backend
os.environ['USE_TORCH'] = '1'  # Force PyTorch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import re
import time
import json
import hashlib
import numpy as np
import warnings

import matplotlib.pyplot as plt


# NLTK for sentence splitting
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Core ML/DS libraries
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from kneed import KneeLocator

# Suppress irrelevant warnings from scikit-learn/numpy
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # to suppress warnings

import seaborn as sns 

# NEW: Import for pairwise distance matrix (for Wasserstein) and ot for the calculation
from scipy.spatial.distance import cdist
import ot

from DIB_with_KL_upper_bound import DIBAnalyzer

class SemanticMutualInformationAnalyzer:
    """
    Implements the Semantic Mutual Information (SMI) method for hallucination detection.
    (Version 3: Includes plotting and result summarization)
    """
    def __init__(self,
                 llm_client: OpenAI,
                 embedding_model: SentenceTransformer,
                 llm_model_name: str,
                 embedding_model_name: str,
                 cache_dir: str = "smi_cache"):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"SMI Analyzer initialized with LLM='{self.llm_model_name}', Embedding='{self.embedding_model_name}'")

    # --- Data generation and utility functions (no changes) ---
    def _get_cache_prefix(self, initial_prompt: str, num_paraphrases: int, num_samples_per_prompt: int) -> str:
        unique_string = f"{initial_prompt}-{num_paraphrases}-{num_samples_per_prompt}-{self.llm_model_name}-{self.embedding_model_name}"
        hash_id = hashlib.sha1(unique_string.encode()).hexdigest()[:10]
        return os.path.join(self.cache_dir, f"run_{hash_id}")

    def _generate_recursive_paraphrases(self, initial_prompt: str, num_paraphrases: int) -> list[str]:
        print(f"Generating {num_paraphrases} paraphrases...")
        # ... (implementation is unchanged)
        prompts = [initial_prompt]
        current_prompt = initial_prompt
        meta_prompt_template_prev = (
            "You are an expert assistant that paraphrases text for linguistic experiments.\n"
            "The user will provide text that has two parts: ## CONSTRAINTS ## and ## CONTENT ##.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST copy the ## CONSTRAINTS ## part VERBATIM and place it at the beginning of your output.\n"
            "2. You MUST rephrase the ## CONTENT ## part to have the same semantic meaning but with different wording.\n"
            "3. Any final instructions (like 'End your output with...') must also be copied VERBATIM.\n"
            "4. Only return the final, combined text. Do not add commentary.\n\n"
            "--- USER'S TEXT ---\n"
            "\"{prompt_with_partitions}\""
        )

        # use the same metaprompt template as before for the CMTF:
        meta_prompt_template = (
            "You are an expert assistant that paraphrases text for linguistic experiments.\n"
            "The user will provide text that has two parts: ## CONSTRAINTS ## and ## CONTENT ##.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST copy the ## CONSTRAINTS ## part VERBATIM and place it at the beginning of your output.\n"
            "2. You MUST rephrase the ## CONTENT ## part to have the same semantic meaning but with different wording.\n"

            "3. Any final instructions (like 'End your output with...') must also be copied VERBATIM.\n"
            "4. Only return the final, combined text. Do not add commentary.\n\n"
            "--- USER'S TEXT ---\n"
            "{prompt_with_partitions}"
        )    
        for i in range(num_paraphrases - 1):
            try:
                prompt_parts = current_prompt.split('\n', 1)
                constraints_part = prompt_parts[0]
                content_part = prompt_parts[1] if len(prompt_parts) > 1 else ""
                partitioned_prompt = (
                    f"## CONSTRAINTS ##\n{constraints_part}\n\n"
                    f"## CONTENT ##\n{content_part}"
                )
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": meta_prompt_template.format(prompt_with_partitions=partitioned_prompt)}],
                    temperature=0.7,
                )
                paraphrase = response.choices[0].message.content.strip()
                prompts.append(paraphrase)
                current_prompt = paraphrase
                time.sleep(1)
            except Exception as e:
                print(f"Error generating paraphrase {i+1}: {e}")
                prompts.append(current_prompt)
        return prompts

    def _generate_answers(self, prompts: list[str], num_samples_per_prompt: int) -> list[list[str]]:
        print(f"Generating {num_samples_per_prompt} answers for each of {len(prompts)} prompts...")
        # ... (implementation is unchanged)
        all_answers = []
        for i, prompt in enumerate(prompts):
            prompt_answers = []
            for _ in range(num_samples_per_prompt):
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8
                    )
                    prompt_answers.append(response.choices[0].message.content)
                except Exception as e:
                    print(f"Error on prompt {i}: {e}")
                    prompt_answers.append("GENERATION_ERROR")
            all_answers.append(prompt_answers)
            time.sleep(1)
        return all_answers

    def _split_into_sentences(self, text: str) -> list[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        return [s for s in sentences if s]

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, show_progress_bar=True, batch_size=8)  # Reduced for MPS memory

    # --- NEW PLOTTING FUNCTION FOR ELBOW METHOD ---
    def _plot_elbow_curve(self, k_values, inertias, elbow_k, run_prefix):
        """Generates and saves the elbow plot for cluster analysis."""
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertias, 'bo-', markersize=8)
        if elbow_k:
            plt.vlines(elbow_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='r',
                       label=f'Optimal k = {elbow_k}')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
        plt.title("Elbow Method for Optimal Number of Topics")
        plt.legend()
        plt.grid(True, linestyle=':')
        
        file_path = f"{run_prefix}_elbow_plot.png"
        # plt.savefig(file_path)
        # print(f"Elbow plot saved to: {file_path}")
        plt.close() # Close plot to prevent it from displaying inline in some environments

    def _estimate_optimal_clusters_elbow(self, embeddings: np.ndarray, run_prefix: str, k_range: tuple = (2, 10)) -> int:
        if embeddings.shape[0] < k_range[0]: return 1
        inertias = []
        k_values = list(range(k_range[0], k_range[1] + 1))
        for k in k_values:
            if embeddings.shape[0] <= k:
                k_values = k_values[:len(inertias)]; break
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(embeddings)
            inertias.append(kmeans.inertia_)

        if len(k_values) < 2: return 1
        kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
        elbow_k = kneedle.elbow if kneedle.elbow else k_values[0]
        self._plot_elbow_curve(k_values, inertias, elbow_k, run_prefix)
        return elbow_k
        


    # --- NEW: DHILLON IT CO-CLUSTERING IMPLEMENTATION ---
    def _kl_divergence(self, p, q):
        """Calculates KL divergence D(p || q) for discrete distributions."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        # Add a small epsilon to avoid log(0) and division by zero
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))
        






        
    def analyze(self, initial_prompt: str, num_paraphrases: int, num_samples_per_prompt: int,
            k_range: tuple = (2, 8), weights: dict = None,
            normalization_factors: dict = None, num_clusters = None,
            alpha: float = 0.05, max_iters: int = 50,
            convergence_threshold: float = 1e-3,
            use_DIB = True, context: str = None) -> dict:
        """
        Analyzes semantic distance metrics for prompt-answer pairs or question-context-answer triplets.

        Args:
            initial_prompt: The initial question/prompt text
            num_paraphrases: Number of paraphrases to generate
            num_samples_per_prompt: Number of answer samples per prompt
            k_range: Range for optimal cluster estimation
            weights: Weights for metric combination
            normalization_factors: Normalization factors for metrics
            num_clusters: Fixed number of clusters (optional)
            alpha: Alpha parameter for ITCC refinement
            max_iters: Maximum iterations for ITCC
            convergence_threshold: Convergence threshold for ITCC
            use_DIB: Whether to use DIB clustering (default: True)
            context: Optional context text (for QCA triplet analysis)

        Returns:
            Dictionary containing all metrics and results
        """
        if weights is None:
            weights = {'w_entropy': 0.3, 'w_wasserstein': 0.2, 'w_jsd': 0.5}
        if normalization_factors is None:
            normalization_factors = {'max_ed': 1.0, 'max_wd': 1.0, 'max_jsd': 1.0}
    
        # --- 1. Data Generation and Caching ---
        run_prefix = self._get_cache_prefix(initial_prompt, num_paraphrases, num_samples_per_prompt)
        prompts_file, answers_file = f"{run_prefix}_prompts.json", f"{run_prefix}_answers.json"
        
        if os.path.exists(prompts_file) and os.path.exists(answers_file):
            print(f"Loading cached text data from '{run_prefix}_*'")
            with open(prompts_file, 'r') as f: prompts = json.load(f)
            with open(answers_file, 'r') as f: answers = json.load(f)
        else:
            prompts = self._generate_recursive_paraphrases(initial_prompt, num_paraphrases)
            answers = self._generate_answers(prompts, num_samples_per_prompt)
            # This logic was already here and is correct for saving the raw prompts/answers
            with open(prompts_file, 'w') as f: json.dump(prompts, f, indent=2)
            with open(answers_file, 'w') as f: json.dump(answers, f, indent=2)
    
        # --- 2. Sentence Splitting and Aggregation ---
        prompt_sentences_by_prompt = [self._split_into_sentences(p) for p in prompts]
        answer_sentences_by_prompt_set = [[s for a in p_ans for s in self._split_into_sentences(a)] for p_ans in answers]

        all_prompt_sentences = [s for group in prompt_sentences_by_prompt for s in group]
        all_answer_sentences = [s for group in answer_sentences_by_prompt_set for s in group]

        # --- Process Context if provided (for QCA triplet analysis) ---
        has_context = context is not None and len(context.strip()) > 0
        if has_context:
            context_sentences = self._split_into_sentences(context)
            all_context_sentences = context_sentences
            print(f"Context provided: {len(all_context_sentences)} sentences extracted")
            all_sentences = all_prompt_sentences + all_context_sentences + all_answer_sentences
        else:
            all_context_sentences = []
            all_sentences = all_prompt_sentences + all_answer_sentences

        # Save the final flat list of all sentences to a JSON file.
        all_sentences_file = f"{run_prefix}_all_sentences.json"
        try:
            with open(all_sentences_file, 'w') as f:
                json.dump(all_sentences, f, indent=2)
            print(f"Successfully saved {len(all_sentences)} combined sentences to '{all_sentences_file}'")
        except Exception as e:
            print(f"Error saving sentences to JSON: {e}")

        if not all_prompt_sentences or not all_answer_sentences: return {}
    
        # --- 3. Embedding Generation with Caching ---
        prompt_embeddings_file = f"{run_prefix}_prompt_embeddings.npy"
        answer_embeddings_file = f"{run_prefix}_answer_embeddings.npy"
        context_embeddings_file = f"{run_prefix}_context_embeddings.npy"

        # Check if we need to load/generate context embeddings
        files_exist = os.path.exists(prompt_embeddings_file) and os.path.exists(answer_embeddings_file)
        if has_context:
            files_exist = files_exist and os.path.exists(context_embeddings_file)

        if files_exist:
            print(f"Loading cached embeddings from '{run_prefix}_*'")
            prompt_embeddings = np.load(prompt_embeddings_file)
            answer_embeddings = np.load(answer_embeddings_file)
            if has_context:
                context_embeddings = np.load(context_embeddings_file)
        else:
            print("No cached embeddings found. Generating and saving embeddings...")
            prompt_embeddings = self._embed_texts(all_prompt_sentences)
            answer_embeddings = self._embed_texts(all_answer_sentences)
            np.save(prompt_embeddings_file, prompt_embeddings)
            np.save(answer_embeddings_file, answer_embeddings)
            if has_context:
                context_embeddings = self._embed_texts(all_context_sentences)
                np.save(context_embeddings_file, context_embeddings)
            print("Cached embeddings for future runs.")

        print("Performing joint clustering...")
        if has_context:
            all_embeddings = np.vstack([prompt_embeddings, context_embeddings, answer_embeddings])
            print(f"Clustering {len(all_embeddings)} sentences: {len(prompt_embeddings)} prompts, {len(context_embeddings)} context, {len(answer_embeddings)} answers")
        else:
            all_embeddings = np.vstack([prompt_embeddings, answer_embeddings])
            print(f"Clustering {len(all_embeddings)} sentences: {len(prompt_embeddings)} prompts, {len(answer_embeddings)} answers")
      

        np.save("all_embeddings.npy", all_embeddings)
        
        optimal_k = self._estimate_optimal_clusters_elbow(all_embeddings, run_prefix, k_range)
        print(f"Optimal k estimated as: {optimal_k}")
    
        # clustering = AgglomerativeClustering(n_clusters=optimal_k)


        if use_DIB == False:
            # Use geometric clustering
        
            # try this instead:
            if num_clusters:
                clustering = AgglomerativeClustering(n_clusters=num_clusters)
            
            else:
                # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0)
                clustering = AgglomerativeClustering(n_clusters=optimal_k)

            all_labels = clustering.fit_predict(all_embeddings)

        elif use_DIB == True:
            # === Step 1: Define DIB Hyperparameters (no change here) ===
            max_n_clusters = 15 # Set a reasonable upper bound
            tau_values = np.logspace(-2, 0, 50)  # A good, wide sweep
            print('Tau values for sweep:', tau_values)
           
            # === Step 2: Run the DIB Analysis (no change here) ===
            # We need to pass the sentences to the analyzer now
            all_sentences = all_prompt_sentences + all_answer_sentences
            analyzer = DIBAnalyzer(all_embeddings, all_sentences)
            
            print("Starting DIB analysis to find optimal topics...")
            analyzer.run(tau_values=tau_values, max_n_clusters=max_n_clusters)
            
            # ==================== MODIFICATION STARTS HERE ====================
            
            # === Step 3: Get the Final Recommended Clustering ===
            # Instead of calling the old get_recommendation, we call the new one.
            # We can also set a minimum number of clusters to consider.
            final_recommendation = analyzer.get_final_recommendation(min_clusters=3)
            
            if final_recommendation:
                # Extract the final cluster assignments and the discovered optimal k
                all_labels = final_recommendation['assignments']
                optimal_k_discovered = final_recommendation['n_clusters']
                
                print(f"\nDIB final analysis complete. Using optimal nc = {optimal_k_discovered}")
                
                # OPTIONAL: You can also run the topic analysis here for immediate feedback
                analyzer.analyze_cluster_topics(final_recommendation)
                
            else:
                # Handle the case where the analysis might not find a stable solution
                print("\nCRITICAL WARNING: DIB analysis could not find a stable clustering solution.")
                print("Falling back to elbow method on geometric clustering.")
                # As a fallback, you can revert to the non-DIB method
                optimal_k = self._estimate_optimal_clusters_elbow(all_embeddings, run_prefix, k_range)
                clustering = AgglomerativeClustering(n_clusters=optimal_k)
                all_labels = clustering.fit_predict(all_embeddings)

            # ===================== MODIFICATION ENDS HERE =====================

        final_num_clusters = len(np.unique(all_labels))

        # Extract labels for each component (Q, C if present, A)
        prompt_labels_flat = all_labels[:len(all_prompt_sentences)]
        if has_context:
            context_labels_flat = all_labels[len(all_prompt_sentences):len(all_prompt_sentences) + len(all_context_sentences)]
            answer_labels_flat = all_labels[len(all_prompt_sentences) + len(all_context_sentences):]
        else:
            context_labels_flat = None
            answer_labels_flat = all_labels[len(all_prompt_sentences):]

        print("Calculating all metrics in a single pass...")

        # --- 2. GLOBAL (Aggregate-First) METRICS ---

        prompt_agg_counts = np.bincount(prompt_labels_flat, minlength=final_num_clusters)
        prompt_dist_global = prompt_agg_counts / np.sum(prompt_agg_counts) if np.sum(prompt_agg_counts) > 0 else np.zeros(final_num_clusters)

        # Calculate context distribution if context is provided (similar to prompt distribution)
        if has_context:
            context_agg_counts = np.bincount(context_labels_flat, minlength=final_num_clusters)
            context_dist_global = context_agg_counts / np.sum(context_agg_counts) if np.sum(context_agg_counts) > 0 else np.zeros(final_num_clusters)
        else:
            context_dist_global = None

        answer_agg_counts = np.bincount(answer_labels_flat, minlength=final_num_clusters)
        answer_dist_global = answer_agg_counts / np.sum(answer_agg_counts) if np.sum(answer_agg_counts) > 0 else np.zeros(final_num_clusters)

        # Compute divergences between Q and A (always), and between C and A (if context present)
        global_jsd = jensenshannon(prompt_dist_global, answer_dist_global, base=2)
        if has_context:
            global_jsd_context_answer = jensenshannon(context_dist_global, answer_dist_global, base=2)
            global_jsd_prompt_context = jensenshannon(prompt_dist_global, context_dist_global, base=2)
        else:
            global_jsd_context_answer = None
            global_jsd_prompt_context = None
        entropy_diff_abs = abs(entropy(answer_dist_global, base=2) - entropy(prompt_dist_global, base=2))
        
        prompt_centroid = np.mean(prompt_embeddings, axis=0)
        answer_centroid = np.mean(answer_embeddings, axis=0)
        
        # w_dist = np.linalg.norm(prompt_centroid - answer_centroid) # not needed

        # --- THIS BLOCK REPLACES THE OLD 'w_dist' CALCULATION ---
        # Wasserstein Distance (Exact Earth Mover's Distance)
        print("Calculating exact Wasserstein distance...")
        n_prompts = prompt_embeddings.shape[0]
        n_answers = answer_embeddings.shape[0]
        
        # The distributions are uniform over the set of sentence embeddings
        prompt_weights = np.ones(n_prompts) / n_prompts
        answer_weights = np.ones(n_answers) / n_answers
        
        # Compute the pairwise cost matrix (squared Euclidean distance between all sentence pairs)
        cost_matrix = cdist(prompt_embeddings, answer_embeddings, 'sqeuclidean')
        
        # Compute the exact Earth Mover's Distance
        # ot.emd2 returns the squared optimal transport cost, so we take the square root
        # to get the 1-Wasserstein distance.
        wasserstein_dist = np.sqrt(ot.emd2(prompt_weights, answer_weights, cost_matrix))
    

        # s_hallucination = (weights['w_entropy'] * entropy_diff_abs + weights['w_wasserstein'] * w_dist + weights['w_jsd'] * global_jsd)
        
        epsilon = 1e-12
        global_kl_pq = entropy(prompt_dist_global + epsilon, answer_dist_global + epsilon, base=2)
        global_kl_qp = entropy(answer_dist_global + epsilon, prompt_dist_global + epsilon, base=2)

        # Calculate global entropies once for reuse
        H_p_global = entropy(prompt_dist_global, base=2)
        H_a_global = entropy(answer_dist_global, base=2)

        # Calculate context-specific metrics if context is provided
        if has_context:
            H_c_global = entropy(context_dist_global, base=2)
            global_kl_ca = entropy(context_dist_global + epsilon, answer_dist_global + epsilon, base=2)
            global_kl_ac = entropy(answer_dist_global + epsilon, context_dist_global + epsilon, base=2)
            global_kl_pc = entropy(prompt_dist_global + epsilon, context_dist_global + epsilon, base=2)
            global_kl_cp = entropy(context_dist_global + epsilon, prompt_dist_global + epsilon, base=2)

            # Compute Semantic Faithfulness (SF) and Semantic Entropy Production (SEP) metrics
            faithfulness_results = self.compute_faithfulness_metrics(
                context_dist_global, prompt_dist_global, answer_dist_global
            )
        else:
            H_c_global = None
            global_kl_ca = None
            global_kl_ac = None
            global_kl_pc = None
            global_kl_cp = None
            faithfulness_results = None
    
    
        # --- 3. ENSEMBLE AND OTHER DIAGNOSTIC METRICS (Single Efficient Loop) ---
        local_jsds, local_kls_pq, local_kls_qp = [], [], []
        local_prob_matrices, local_conditional_entropies = [], []
        # global_joint_counts = np.zeros((optimal_k, optimal_k))
        # --- THIS IS THE CORRECTED LINE FOR THE GLOBAL TABLE ---
        global_joint_counts = np.zeros((final_num_clusters, final_num_clusters))
        prompt_assignments, answer_assignments = [], []
        prompt_sent_idx, answer_sent_idx = 0, 0
        
        for i in range(num_paraphrases):
            num_prompt_sents, num_answer_sents = len(prompt_sentences_by_prompt[i]), len(answer_sentences_by_prompt_set[i])
            
            prompt_subset_labels = prompt_labels_flat[prompt_sent_idx:prompt_sent_idx+num_prompt_sents]
            answer_subset_labels = answer_labels_flat[answer_sent_idx:answer_sent_idx+num_answer_sents]
            
            prompt_assignments.append(np.bincount(prompt_subset_labels).argmax() if num_prompt_sents > 0 else -1)
            answer_assignments.append(np.bincount(answer_subset_labels).argmax() if num_answer_sents > 0 else -1)
            
            if num_prompt_sents > 0 and num_answer_sents > 0:

                # Create local distributions FOR THE CORRECT SIZE
                p_counts_local = np.bincount(prompt_subset_labels, minlength=final_num_clusters)
                p_dist_local = p_counts_local / np.sum(p_counts_local)
                a_counts_local = np.bincount(answer_subset_labels, minlength=final_num_clusters)
                a_dist_local = a_counts_local / np.sum(a_counts_local)
            
                
                # p_counts_local = np.bincount(prompt_subset_labels, minlength=optimal_k)
                # p_dist_local = p_counts_local / np.sum(p_counts_local)
                # a_counts_local = np.bincount(answer_subset_labels, minlength=optimal_k)
                # a_dist_local = a_counts_local / np.sum(a_counts_local)
                
                local_jsds.append(jensenshannon(p_dist_local, a_dist_local, base=2))
                local_kls_pq.append(entropy(p_dist_local + epsilon, a_dist_local + epsilon, base=2))
                local_kls_qp.append(entropy(a_dist_local + epsilon, p_dist_local + epsilon, base=2))
                
                # local_counts = np.zeros((optimal_k, optimal_k))
                local_counts = np.zeros((final_num_clusters, final_num_clusters))
                for p_label in prompt_subset_labels:
                    for a_label in answer_subset_labels:
                        local_counts[p_label, a_label] += 1
                global_joint_counts += local_counts
                local_prob_matrices.append(local_counts / np.sum(local_counts))
                
                local_joint_prob = local_counts / np.sum(local_counts)
                local_p_x = np.sum(local_joint_prob, axis=1)
                h_y_given_x = 0.0
                for p_idx in range(final_num_clusters): # range(optimal_k):
                    if local_p_x[p_idx] > epsilon:
                        p_y_given_x = local_joint_prob[p_idx, :] / local_p_x[p_idx]
                        h_y_given_x += local_p_x[p_idx] * entropy(p_y_given_x, base=2)
                local_conditional_entropies.append(h_y_given_x)
                
            prompt_sent_idx += num_prompt_sents
            answer_sent_idx += num_answer_sents
    
        ensemble_jsd = np.mean(local_jsds) if local_jsds else 0.0
        ensemble_kl_pq = np.mean(local_kls_pq) if local_kls_pq else 0.0
        ensemble_kl_qp = np.mean(local_kls_qp) if local_kls_qp else 0.0
        
        averaged_prob_matrix = np.mean(local_prob_matrices, axis=0) if local_prob_matrices else np.zeros((optimal_k, optimal_k))
        heatmap_path = self._plot_averaged_cooccurrence_matrix(averaged_prob_matrix, optimal_k, run_prefix)
        mi_averaged, nmi_averaged = 0.0, 0.0
        if np.sum(averaged_prob_matrix) > 0:
            p_x_avg, p_y_avg = np.sum(averaged_prob_matrix, axis=1), np.sum(averaged_prob_matrix, axis=0)
            for i in range(final_num_clusters):
                for j in range(final_num_clusters):
                    if averaged_prob_matrix[i, j] > epsilon and p_x_avg[i] > epsilon and p_y_avg[j] > epsilon:
                        mi_averaged += averaged_prob_matrix[i, j] * np.log2(averaged_prob_matrix[i, j] / (p_x_avg[i] * p_y_avg[j]))
            H_p_avg, H_a_avg = entropy(p_x_avg, base=2), entropy(p_y_avg, base=2)
            nmi_averaged = mi_averaged / min(H_p_avg, H_a_avg) if min(H_p_avg, H_a_avg) > 0 else 0.0
        
        H_Y = entropy(answer_dist_global, base=2)
        H_Y_given_X = np.mean(local_conditional_entropies) if local_conditional_entropies else 0
        mi_ensemble = H_Y - H_Y_given_X
    
        # --- THIS BLOCK WAS MISSING - IT IS NOW RESTORED ---
        mi_global, nmi_global, H_p_global, H_a_global, phi_global = 0.0, 0.0, 0.0, 0.0, float('inf')
        if np.sum(global_joint_counts) > 0:
            joint_prob = global_joint_counts / np.sum(global_joint_counts)
            p_x, p_y = np.sum(joint_prob, axis=1), np.sum(joint_prob, axis=0)
            for i in range(final_num_clusters):
                for j in range(final_num_clusters):
                    if joint_prob[i, j] > epsilon and p_x[i] > epsilon and p_y[j] > epsilon:
                        mi_global += joint_prob[i, j] * np.log2(joint_prob[i, j] / (p_x[i] * p_y[j]))
            H_p_global, H_a_global = entropy(p_x, base=2), entropy(p_y, base=2)
            nmi_global = mi_global / min(H_p_global, H_a_global) if min(H_p_global, H_a_global) > 0 else 0.0
            phi_global = (H_a_global - mi_global) / H_p_global if H_p_global > 0 else float('inf')
        
        valid_indices = [i for i, (p, a) in enumerate(zip(prompt_assignments, answer_assignments)) if p != -1 and a != -1]
        ami_score = adjusted_mutual_info_score([prompt_assignments[i] for i in valid_indices], [answer_assignments[i] for i in valid_indices]) if valid_indices else 0.0
    
        answers_by_prompt = [ans_set for ans_set in answers]
        se_metrics = self._calculate_semantic_entropy(answers_by_prompt, run_prefix, k_range)

        # # the final S_H score should use ensemble jsd
        s_hallucination = (weights['w_entropy'] * entropy_diff_abs + 
                       weights['w_wasserstein'] * wasserstein_dist + 
                       weights['w_jsd'] * ensemble_jsd) # <-- CORRECTED VARIABLE

        # --- [NEW BLOCK] SCALED METRIC CALCULATION ---
        # We now normalize the key scores by the prompt's global entropy (H_p_global)
        # to create complexity-adjusted versions.
        if H_p_global > 1e-9:
            s_h_scaled = s_hallucination / H_p_global
            ensemble_kl_qp_scaled = ensemble_kl_qp / H_p_global
        else:
            # If prompt has zero entropy, any divergence/exploration is infinite
            s_h_scaled = float('inf')
            ensemble_kl_qp_scaled = float('inf')

    
        # --- 4. FINAL RESULTS DICTIONARY ---
        results = {
            "run_prefix": run_prefix, "initial_prompt": initial_prompt,
            "has_context": has_context,
            "sdm_hallucination_score": s_h_scaled, # New scaled score # s_hallucination,
            "phi_hallucination_indicator": phi_global,
            "metrics": {
                "optimal_k": final_num_clusters, # Use the confirmed number of clusters # optimal_k,
                # Global Metrics (Q-A)
                "global_jensen_shannon_divergence": global_jsd,
                "global_kl_divergence_prompt_answer": global_kl_pq,
                "global_kl_divergence_answer_prompt": global_kl_qp,
                "entropy_difference": entropy_diff_abs,
                "global_prompt_entropy": H_p_global,
                "global_answer_entropy": H_a_global,
                "wasserstein_distance": wasserstein_dist,
                # Context Metrics (QCA triplet - only if context provided)
                "global_context_entropy": H_c_global,
                "global_jsd_context_answer": global_jsd_context_answer,
                "global_jsd_prompt_context": global_jsd_prompt_context,
                "global_kl_divergence_context_answer": global_kl_ca,
                "global_kl_divergence_answer_context": global_kl_ac,
                "global_kl_divergence_prompt_context": global_kl_pc,
                "global_kl_divergence_context_prompt": global_kl_cp,
                # Semantic Faithfulness and Entropy Production Metrics (QCA triplet only)
                "semantic_faithfulness": faithfulness_results['F_S'] if faithfulness_results else None,
                "semantic_faithfulness_D_min": faithfulness_results['D_min'] if faithfulness_results else None,
                "semantic_faithfulness_iterations": faithfulness_results['iterations'] if faithfulness_results else None,
                "sep_system_entropy_change": faithfulness_results['S_dot_system'] if faithfulness_results else None,
                "sep_total_entropy_production": faithfulness_results['S_dot_total_approx'] if faithfulness_results else None,
                # Ensemble Metrics
                "ensemble_jensen_shannon_divergence": ensemble_jsd,
                "ensemble_kl_divergence_prompt_answer": ensemble_kl_pq,
                "ensemble_kl_divergence_answer_prompt": ensemble_kl_qp_scaled, # report the scaled version # ensemble_kl_qp,
                "ensemble_mutual_information": mi_ensemble,
                # Other Diagnostics
                "averaged_mutual_information": mi_averaged,
                "adjusted_mutual_information": ami_score,
                "global_mutual_information": mi_global, # Included for completeness
                # Baselines
                "semantic_entropy_original": se_metrics["semantic_entropy_original"],
                "mean_semantic_entropy": se_metrics["mean_semantic_entropy"],
                # Path
                "averaged_cooccurrence_heatmap_path": heatmap_path
            },
            "raw_data": {"prompts": prompts, "answers": answers, "context": context if has_context else None}
        }
        return results

    def analyze_triplets(self, prompt_texts: list, answer_texts: list,
                        context_texts: list = None,
                        num_clusters: int = 10,
                        k_range: tuple = (2, 8),
                        weights: dict = None,
                        use_DIB: bool = True,
                        cache_prefix: str = "qca_triplets") -> dict:
        """
        Analyzes pre-made QCA (Question-Context-Answer) triplets using SDM metrics.

        This method is designed for datasets where you already have:
        - Multiple prompt/question variations (Q)
        - Corresponding answers (A)
        - A shared context document (C)

        Unlike analyze() which generates paraphrases and answers, this method
        accepts your pre-made data and performs semantic distance analysis.

        Args:
            prompt_texts: List of N prompt/question texts (one per variation)
            answer_texts: List of N answer texts (one per prompt)
            context_texts: List of N context texts (same for all, or per-prompt variations)
            num_clusters: Number of clusters for topic modeling (default: 10)
            k_range: Range for optimal cluster estimation if use_DIB=False
            weights: Weights for metric combination
            use_DIB: Whether to use DIB clustering (default: True)
            cache_prefix: Prefix for cached embeddings

        Returns:
            Dictionary containing all SDM metrics including SF/SEP scores
        """
        if weights is None:
            weights = {'w_entropy': 0.3, 'w_wasserstein': 0.2, 'w_jsd': 0.5}

        num_triplets = len(prompt_texts)
        if len(answer_texts) != num_triplets:
            raise ValueError(f"Number of answers ({len(answer_texts)}) must match number of prompts ({num_triplets})")

        # Handle context: if single string, replicate for all triplets
        if context_texts is None:
            has_context = False
            context_texts = [None] * num_triplets
        elif isinstance(context_texts, str):
            # Single context string provided - use for all
            has_context = True
            context_texts = [context_texts] * num_triplets
        else:
            # List of contexts provided
            has_context = True
            if len(context_texts) != num_triplets:
                raise ValueError(f"Number of contexts ({len(context_texts)}) must match number of prompts ({num_triplets})")

        print(f"Analyzing {num_triplets} QCA triplets with context={has_context}")

        # --- 1. Sentence Splitting ---
        print("Splitting texts into sentences...")
        prompt_sentences_by_prompt = [self._split_into_sentences(p) for p in prompt_texts]
        answer_sentences_by_prompt_set = [self._split_into_sentences(a) for a in answer_texts]

        all_prompt_sentences = [s for group in prompt_sentences_by_prompt for s in group]
        all_answer_sentences = [s for group in answer_sentences_by_prompt_set for s in group]

        if has_context:
            # Extract unique context sentences (in case same context repeated)
            context_sentences_by_context = [self._split_into_sentences(c) for c in context_texts if c]
            # Use first context for global analysis (assuming same context for all)
            all_context_sentences = context_sentences_by_context[0] if context_sentences_by_context else []
            all_sentences = all_prompt_sentences + all_context_sentences + all_answer_sentences
            print(f"Sentences: {len(all_prompt_sentences)} prompts, {len(all_context_sentences)} context, {len(all_answer_sentences)} answers")
        else:
            all_context_sentences = []
            all_sentences = all_prompt_sentences + all_answer_sentences
            print(f"Sentences: {len(all_prompt_sentences)} prompts, {len(all_answer_sentences)} answers")

        if not all_prompt_sentences or not all_answer_sentences:
            raise ValueError("No sentences extracted from prompts or answers")

        # --- 2. Embedding Generation ---
        print("Generating embeddings...")
        prompt_embeddings = self._embed_texts(all_prompt_sentences)
        answer_embeddings = self._embed_texts(all_answer_sentences)

        if has_context:
            context_embeddings = self._embed_texts(all_context_sentences)
            all_embeddings = np.vstack([prompt_embeddings, context_embeddings, answer_embeddings])
        else:
            all_embeddings = np.vstack([prompt_embeddings, answer_embeddings])

        print(f"Generated {len(all_embeddings)} embeddings")

        # --- 3. Clustering ---
        print("Performing joint clustering...")

        if use_DIB:
            # DIB clustering - allow more clusters for better semantic diversity
            max_n_clusters = 25  # Increased from 15
            tau_values = np.logspace(-2, 1, 50)  # Wider range to explore more cluster counts

            analyzer = DIBAnalyzer(all_embeddings, all_sentences)
            print("Running DIB analysis...")
            analyzer.run(tau_values=tau_values, max_n_clusters=max_n_clusters)

            final_recommendation = analyzer.get_final_recommendation(min_clusters=8)  # Increased from 3

            if final_recommendation:
                all_labels = final_recommendation['assignments']
                optimal_k = final_recommendation['n_clusters']
                print(f"DIB complete. Using optimal k={optimal_k}")
            else:
                print("DIB failed. Falling back to geometric clustering")
                clustering = KMeans(n_clusters=num_clusters, random_state=42)
                all_labels = clustering.fit_predict(all_embeddings)
                optimal_k = num_clusters
        else:
            # Geometric clustering
            clustering = KMeans(n_clusters=num_clusters, random_state=42)
            all_labels = clustering.fit_predict(all_embeddings)
            optimal_k = num_clusters

        final_num_clusters = len(np.unique(all_labels))
        print(f"Clustering complete. Final clusters: {final_num_clusters}")

        # --- 4. Extract Labels ---
        prompt_labels_flat = all_labels[:len(all_prompt_sentences)]
        if has_context:
            context_labels_flat = all_labels[len(all_prompt_sentences):len(all_prompt_sentences) + len(all_context_sentences)]
            answer_labels_flat = all_labels[len(all_prompt_sentences) + len(all_context_sentences):]
        else:
            context_labels_flat = None
            answer_labels_flat = all_labels[len(all_prompt_sentences):]

        # --- 5. Compute Global Distributions ---
        print("Computing semantic distance metrics...")

        # Entropy regularization constant - ensures all clusters get non-zero probability
        eps_H = 1e-6

        # Compute regularized probability distributions
        # P(topic|Q) = (counts + eps_H) / (sum(counts) + eps_H * K)
        prompt_agg_counts = np.bincount(prompt_labels_flat, minlength=final_num_clusters)
        prompt_dist_global = (prompt_agg_counts + eps_H) / (np.sum(prompt_agg_counts) + eps_H * final_num_clusters)

        answer_agg_counts = np.bincount(answer_labels_flat, minlength=final_num_clusters)
        answer_dist_global = (answer_agg_counts + eps_H) / (np.sum(answer_agg_counts) + eps_H * final_num_clusters)

        if has_context:
            context_agg_counts = np.bincount(context_labels_flat, minlength=final_num_clusters)
            context_dist_global = (context_agg_counts + eps_H) / (np.sum(context_agg_counts) + eps_H * final_num_clusters)
        else:
            context_dist_global = None

        # --- 6. Compute Divergences and Distances ---
        # Note: distributions already have eps_H regularization, so additional epsilon is optional
        epsilon = eps_H  # Use same epsilon for extra numerical safety

        # Q-A divergences
        global_jsd = jensenshannon(prompt_dist_global, answer_dist_global, base=2)
        global_kl_pq = entropy(prompt_dist_global + epsilon, answer_dist_global + epsilon, base=2)
        global_kl_qp = entropy(answer_dist_global + epsilon, prompt_dist_global + epsilon, base=2)
        H_p_global = entropy(prompt_dist_global, base=2)
        H_a_global = entropy(answer_dist_global, base=2)
        entropy_diff_abs = abs(H_a_global - H_p_global)

        # Wasserstein distance
        prompt_centroid = np.mean(prompt_embeddings, axis=0)
        answer_centroid = np.mean(answer_embeddings, axis=0)
        prompt_weights = prompt_dist_global
        answer_weights = answer_dist_global

        # Build cost matrix
        prompt_cluster_centers = np.array([np.mean(prompt_embeddings[prompt_labels_flat == k], axis=0)
                                          if np.sum(prompt_labels_flat == k) > 0 else np.zeros(self.embedding_dim)
                                          for k in range(final_num_clusters)])
        answer_cluster_centers = np.array([np.mean(answer_embeddings[answer_labels_flat == k], axis=0)
                                          if np.sum(answer_labels_flat == k) > 0 else np.zeros(self.embedding_dim)
                                          for k in range(final_num_clusters)])
        cost_matrix = cdist(prompt_cluster_centers, answer_cluster_centers, metric='euclidean')
        wasserstein_dist = np.sqrt(ot.emd2(prompt_weights, answer_weights, cost_matrix))

        # C-A divergences (if context present)
        if has_context:
            H_c_global = entropy(context_dist_global, base=2)
            global_jsd_context_answer = jensenshannon(context_dist_global, answer_dist_global, base=2)
            global_jsd_prompt_context = jensenshannon(prompt_dist_global, context_dist_global, base=2)
            global_kl_ca = entropy(context_dist_global + epsilon, answer_dist_global + epsilon, base=2)
            global_kl_ac = entropy(answer_dist_global + epsilon, context_dist_global + epsilon, base=2)
            global_kl_pc = entropy(prompt_dist_global + epsilon, context_dist_global + epsilon, base=2)
            global_kl_cp = entropy(context_dist_global + epsilon, prompt_dist_global + epsilon, base=2)

            # Compute SF/SEP metrics
            faithfulness_results = self.compute_faithfulness_metrics(
                context_dist_global, prompt_dist_global, answer_dist_global
            )
        else:
            H_c_global = None
            global_jsd_context_answer = None
            global_jsd_prompt_context = None
            global_kl_ca = None
            global_kl_ac = None
            global_kl_pc = None
            global_kl_cp = None
            faithfulness_results = None

        # --- 7. Ensemble Metrics (per-triplet analysis) ---
        local_jsds, local_kls_pq, local_kls_qp = [], [], []
        local_conditional_entropies = []
        global_joint_counts = np.zeros((final_num_clusters, final_num_clusters))
        prompt_sent_idx, answer_sent_idx = 0, 0

        for i in range(num_triplets):
            num_prompt_sents = len(prompt_sentences_by_prompt[i])
            num_answer_sents = len(answer_sentences_by_prompt_set[i])

            prompt_subset_labels = prompt_labels_flat[prompt_sent_idx:prompt_sent_idx+num_prompt_sents]
            answer_subset_labels = answer_labels_flat[answer_sent_idx:answer_sent_idx+num_answer_sents]

            if num_prompt_sents > 0 and num_answer_sents > 0:
                p_counts_local = np.bincount(prompt_subset_labels, minlength=final_num_clusters)
                p_dist_local = p_counts_local / np.sum(p_counts_local)
                a_counts_local = np.bincount(answer_subset_labels, minlength=final_num_clusters)
                a_dist_local = a_counts_local / np.sum(a_counts_local)

                local_jsds.append(jensenshannon(p_dist_local, a_dist_local, base=2))
                local_kls_pq.append(entropy(p_dist_local + epsilon, a_dist_local + epsilon, base=2))
                local_kls_qp.append(entropy(a_dist_local + epsilon, p_dist_local + epsilon, base=2))

                local_counts = np.zeros((final_num_clusters, final_num_clusters))
                for p_label in prompt_subset_labels:
                    for a_label in answer_subset_labels:
                        local_counts[p_label, a_label] += 1
                global_joint_counts += local_counts

                local_joint_prob = local_counts / np.sum(local_counts)
                local_p_x = np.sum(local_joint_prob, axis=1)
                h_y_given_x = 0.0
                for p_idx in range(final_num_clusters):
                    if local_p_x[p_idx] > epsilon:
                        p_y_given_x = local_joint_prob[p_idx, :] / local_p_x[p_idx]
                        h_y_given_x += local_p_x[p_idx] * entropy(p_y_given_x, base=2)
                local_conditional_entropies.append(h_y_given_x)

            prompt_sent_idx += num_prompt_sents
            answer_sent_idx += num_answer_sents

        ensemble_jsd = np.mean(local_jsds) if local_jsds else 0.0
        ensemble_kl_pq = np.mean(local_kls_pq) if local_kls_pq else 0.0
        ensemble_kl_qp = np.mean(local_kls_qp) if local_kls_qp else 0.0

        H_Y = H_a_global
        H_Y_given_X = np.mean(local_conditional_entropies) if local_conditional_entropies else 0
        mi_ensemble = H_Y - H_Y_given_X

        # Global MI
        mi_global = 0.0
        if np.sum(global_joint_counts) > 0:
            joint_prob = global_joint_counts / np.sum(global_joint_counts)
            p_x = np.sum(joint_prob, axis=1)
            p_y = np.sum(joint_prob, axis=0)
            for i in range(final_num_clusters):
                for j in range(final_num_clusters):
                    if joint_prob[i, j] > epsilon and p_x[i] > epsilon and p_y[j] > epsilon:
                        mi_global += joint_prob[i, j] * np.log2(joint_prob[i, j] / (p_x[i] * p_y[j]))

        # Scaled scores
        if H_p_global > 1e-9:
            s_hallucination = (weights['w_entropy'] * entropy_diff_abs +
                             weights['w_wasserstein'] * wasserstein_dist +
                             weights['w_jsd'] * ensemble_jsd)
            s_h_scaled = s_hallucination / H_p_global
            ensemble_kl_qp_scaled = ensemble_kl_qp / H_p_global
        else:
            s_h_scaled = float('inf')
            ensemble_kl_qp_scaled = float('inf')

        # --- 8. Build Results Dictionary ---
        print("Analysis complete!")

        results = {
            "run_prefix": cache_prefix,
            "num_triplets": num_triplets,
            "has_context": has_context,
            "sdm_hallucination_score": s_h_scaled,
            "metrics": {
                "optimal_k": final_num_clusters,
                # Global Metrics (Q-A)
                "global_jensen_shannon_divergence": global_jsd,
                "global_kl_divergence_prompt_answer": global_kl_pq,
                "global_kl_divergence_answer_prompt": global_kl_qp,
                "entropy_difference": entropy_diff_abs,
                "global_prompt_entropy": H_p_global,
                "global_answer_entropy": H_a_global,
                "wasserstein_distance": wasserstein_dist,
                # Context Metrics (QCA triplet)
                "global_context_entropy": H_c_global,
                "global_jsd_context_answer": global_jsd_context_answer,
                "global_jsd_prompt_context": global_jsd_prompt_context,
                "global_kl_divergence_context_answer": global_kl_ca,
                "global_kl_divergence_answer_context": global_kl_ac,
                "global_kl_divergence_prompt_context": global_kl_pc,
                "global_kl_divergence_context_prompt": global_kl_cp,
                # Semantic Faithfulness and Entropy Production Metrics
                "semantic_faithfulness": faithfulness_results['F_S'] if faithfulness_results else None,
                "semantic_faithfulness_D_min": faithfulness_results['D_min'] if faithfulness_results else None,
                "semantic_faithfulness_iterations": faithfulness_results['iterations'] if faithfulness_results else None,
                "sep_system_entropy_change": faithfulness_results['S_dot_system'] if faithfulness_results else None,
                "sep_total_entropy_production": faithfulness_results['S_dot_total_approx'] if faithfulness_results else None,
                # Ensemble Metrics
                "ensemble_jensen_shannon_divergence": ensemble_jsd,
                "ensemble_kl_divergence_prompt_answer": ensemble_kl_pq,
                "ensemble_kl_divergence_answer_prompt": ensemble_kl_qp_scaled,
                "ensemble_mutual_information": mi_ensemble,
                # Other Diagnostics
                "global_mutual_information": mi_global,
            },
            "distributions": {
                "prompt_dist": prompt_dist_global.tolist(),
                "context_dist": context_dist_global.tolist() if context_dist_global is not None else None,
                "answer_dist": answer_dist_global.tolist(),
                # Short aliases for convenience
                "p_q": prompt_dist_global.tolist(),
                "p_c": context_dist_global.tolist() if context_dist_global is not None else None,
                "p_a": answer_dist_global.tolist()
            },
            "raw_data": {
                "prompts": prompt_texts,
                "answers": answer_texts,
                "context": context_texts[0] if has_context and context_texts else None
            }
        }

        return results

    def compute_semantic_faithfulness(self, p_c, p_q, p_a, Q_init=None,
                                      tol_outer=1e-6, tol_inner=1e-6, max_outer_iter=100):
        """
        Compute Semantic Faithfulness (SF) score using Algorithm 1 from the paper.

        Args:
            p_c: Context probability distribution over topics (numpy array of length N)
            p_q: Question probability distribution over topics (numpy array of length N)
            p_a: Answer probability distribution over topics (numpy array of length N)
            Q_init: Initial guess for Q matrix (N x N). If None, uses uniform initialization.
            tol_outer: Convergence tolerance for outer loop
            tol_inner: Convergence tolerance for inner loop (A-step and Q-step)
            max_outer_iter: Maximum number of outer iterations

        Returns:
            dict: {
                'F_S': Semantic Faithfulness score (0 to 1),
                'D_min': Minimal KL divergence,
                'A_star': Optimal A matrix,
                'Q_star': Optimal Q matrix,
                'iterations': Number of iterations to convergence
            }
        """
        N = len(p_c)
        epsilon = 1e-12

        # Initialize Q matrix if not provided
        if Q_init is None:
            Q = np.ones((N, N)) / N  # Uniform initialization
        else:
            Q = Q_init.copy()

        # Initialize A matrix
        A = np.ones((N, N)) / N

        prev_objective = float('inf')

        for outer_iter in range(max_outer_iter):
            # --- A-STEP: Update A given Q ---
            # Solve fixed-point equation (10) for u_j
            u = np.ones(N)  # Initial guess
            for _ in range(100):  # Inner fixed-point iterations
                u_old = u.copy()
                denominator = np.zeros(N)
                for i in range(N):
                    denominator += p_c[i] * Q[i, :] / (np.sum(Q[i, :] * u) + epsilon)
                u = p_a / (denominator + epsilon)
                if np.linalg.norm(u - u_old) < tol_inner:
                    break

            # Compute A using Eq.(11)
            for i in range(N):
                A[i, :] = Q[i, :] * u / (np.sum(Q[i, :] * u) + epsilon)

            # --- Q-STEP: Update Q given A ---
            # Maximize dual Lagrangian L_2(ξ, ν) using alternating maximization
            xi = np.zeros(N)
            nu = np.zeros(N)

            for _ in range(100):  # Inner alternating maximization
                xi_old = xi.copy()
                nu_old = nu.copy()

                # Update nu with xi fixed
                for i in range(N):
                    sum_term = 0.0
                    for j in range(N):
                        if (nu[i] + xi[j]) > epsilon:
                            sum_term += A[i, j]
                    if sum_term > epsilon:
                        nu[i] = p_c[i] / sum_term

                # Update xi with nu fixed
                for j in range(N):
                    sum_term = 0.0
                    for i in range(N):
                        if (nu[i] + xi[j]) > epsilon:
                            sum_term += p_c[i] * A[i, j] / (nu[i] + xi[j])
                    if sum_term > epsilon:
                        xi[j] = p_q[j] / sum_term

                if np.linalg.norm(xi - xi_old) < tol_inner and np.linalg.norm(nu - nu_old) < tol_inner:
                    break

            # Compute Q using Eq.(13)
            for i in range(N):
                for j in range(N):
                    Q[i, j] = A[i, j] / (nu[i] + xi[j] + epsilon)

            # Normalize Q rows to ensure row-stochastic property
            for i in range(N):
                row_sum = np.sum(Q[i, :])
                if row_sum > epsilon:
                    Q[i, :] /= row_sum

            # --- Compute objective and check convergence ---
            objective = 0.0
            for i in range(N):
                for j in range(N):
                    if A[i, j] > epsilon and Q[i, j] > epsilon:
                        objective += p_c[i] * A[i, j] * np.log(A[i, j] / Q[i, j])

            relative_change = abs(prev_objective - objective) / (abs(prev_objective) + epsilon)

            if relative_change < tol_outer:
                break

            prev_objective = objective

        # Compute final SF score
        D_min = objective
        F_S = 1.0 / (1.0 + D_min)

        return {
            'F_S': F_S,
            'D_min': D_min,
            'A_star': A,
            'Q_star': Q,
            'iterations': outer_iter + 1
        }

    def compute_semantic_entropy_production(self, p_c, p_a, F_S=None, A_star=None):
        """
        Compute Semantic Entropy Production (SEP) metric.

        Args:
            p_c: Context probability distribution over topics
            p_a: Answer probability distribution over topics
            F_S: Semantic Faithfulness score (if available)
            A_star: Optimal A matrix from SF computation (if available)

        Returns:
            dict: {
                'S_dot_system': System entropy change H(p_a) - H(p_c),
                'S_dot_total_approx': Approximate total entropy production,
                'H_context': Entropy of context,
                'H_answer': Entropy of answer
            }
        """
        epsilon = 1e-12

        # Compute marginal entropies
        H_context = entropy(p_c + epsilon, base=2)
        H_answer = entropy(p_a + epsilon, base=2)

        # System entropy change (Eq. 18)
        S_dot_system = H_answer - H_context

        # Approximate total entropy production using Eq.(24)
        if F_S is not None:
            S_dot_total_approx = 1.0 / F_S - 1.0
        else:
            S_dot_total_approx = None

        return {
            'S_dot_system': S_dot_system,
            'S_dot_total_approx': S_dot_total_approx,
            'H_context': H_context,
            'H_answer': H_answer
        }

    def compute_faithfulness_metrics(self, p_c, p_q, p_a, Q_init=None):
        """
        Compute both Semantic Faithfulness (SF) and Semantic Entropy Production (SEP) metrics.

        Args:
            p_c: Context probability distribution over topics
            p_q: Question probability distribution over topics
            p_a: Answer probability distribution over topics
            Q_init: Initial guess for Q matrix (optional)

        Returns:
            dict: Combined results from SF and SEP computations
        """
        # Compute SF score
        sf_results = self.compute_semantic_faithfulness(p_c, p_q, p_a, Q_init=Q_init)

        # Compute SEP using the SF score
        sep_results = self.compute_semantic_entropy_production(
            p_c, p_a,
            F_S=sf_results['F_S'],
            A_star=sf_results['A_star']
        )

        # Combine results
        return {
            **sf_results,
            **sep_results,
            'relationship_check': {
                'high_faithfulness_low_entropy': sf_results['F_S'] > 0.7 and sep_results['S_dot_total_approx'] < 0.5
            }
        }

    def calculate_normalized_scores(self, all_results: list, weights: dict = None) -> list:
        """
        Takes a list of results from multiple 'analyze' runs, normalizes the core
        metrics across the entire set, and computes the final S_H score for each run.
    
        Args:
            all_results (list): A list of dictionary outputs from the 'analyze' method.
            weights (dict): A dictionary of weights for the final score.
    
        Returns:
            list: The original list of results, with the 'smi_hallucination_score' and
                  normalized metrics added to each result's 'metrics' dictionary.
        """
        print("Normalizing metrics and calculating final S_H scores...")
        if not all_results:
            return []
        
        if weights is None:
            weights = {'w_entropy': 0.3, 'w_wasserstein': 0.2, 'w_jsd': 0.5}
    
        # Step 1: Find the maximum value for each metric across all experiments
        max_ed = max(r['metrics']['entropy_difference'] for r in all_results)
        max_wd = max(r['metrics']['wasserstein_distance'] for r in all_results)
        max_jsd = max(r['metrics']['ensemble_jensen_shannon_divergence'] for r in all_results)
    
        # Avoid division by zero if a max is 0 (unlikely but safe)
        max_ed = max_ed if max_ed > 0 else 1.0
        max_wd = max_wd if max_wd > 0 else 1.0
        max_jsd = max_jsd if max_jsd > 0 else 1.0
    
        # Step 2: Iterate through each result, normalize, and calculate the final score
        for result in all_results:
            metrics = result['metrics']
            
            # Normalize the core components to a [0, 1] range
            ed_norm = metrics['entropy_difference'] / max_ed
            wd_norm = metrics['wasserstein_distance'] / max_wd
            jsd_norm = metrics['ensemble_jensen_shannon_divergence'] / max_jsd
            
            # Store the normalized values for transparency
            metrics['normalized_entropy_difference'] = ed_norm
            metrics['normalized_wasserstein_distance'] = wd_norm
            metrics['normalized_ensemble_jsd'] = jsd_norm
            
            # Calculate the final, normalized S_H score
            s_hallucination = (weights['w_entropy'] * ed_norm + 
                               weights['w_wasserstein'] * wd_norm + 
                               weights['w_jsd'] * jsd_norm)
            
            # Add the final score to the result
            result['sdm_hallucination_score'] = s_hallucination
            
        return all_results        
        
    
    def plot_and_summarize_results(self, results: dict, weights: dict = None):
        """
        (GRAND UNIFIED VERSION - CORRECTED)
        This definitive function correctly plots the final score based on ENSEMBLE JSD
        and prints a comprehensive summary of all metrics, with annotations updated
        to reflect the final methodology.
        """
        if weights is None:
            weights = {'w_entropy': 0.4, 'w_wasserstein': 0.2, 'w_jsd': 0.4}
            
        if not results or "metrics" not in results:
            print("No results or metrics found to plot or summarize.")
            return
    
        score = results.get('sdm_hallucination_score', 0.0)
        phi = results.get('phi_hallucination_indicator', float('inf'))
        metrics = results.get('metrics', {})
        run_prefix = results.get('run_prefix', 'run')
        
        # --- 1. PLOTTING THE SCORE BREAKDOWN (CORRECTED to use Ensemble JSD) ---
        components = {
            "Complexity Mismatch\n(Entropy Diff)": weights['w_entropy'] * metrics.get('entropy_difference', 0.0),
            "Distributional Shift\n(Wasserstein)": weights['w_wasserstein'] * metrics.get('wasserstein_distance', 0.0),
            "Semantic Divergence\n(Ensemble JSD)": weights['w_jsd'] * metrics.get('ensemble_jensen_shannon_divergence', 0.0)
        }
    
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(components.keys(), components.values(), color=['skyblue', 'salmon', 'lightgreen'])
        ax.bar_label(bars, fmt='%.4f')
        ax.set_ylabel("Weighted Contribution to Score")
        ax.set_title("Breakdown of SMI Hallucination Score")
        plt.suptitle(f"Total Score (based on Ensemble JSD): {score:.4f}", fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        file_path = f"{run_prefix}_score_breakdown.png"
        plt.savefig(file_path)
        print(f"\nScore breakdown plot saved to: {file_path}")
        plt.close()
        
        # --- 2. COMPREHENSIVE TEXT SUMMARY (CORRECTED Annotations) ---
        print("\n\n" + "="*60)
        print("     GRAND UNIFIED ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nInitial Prompt: {results.get('initial_prompt', 'N/A')}")
        print(f"\n>>> SDM Hallucination Score: {score:.4f} <<<")
        print(f">>> Normalized Conditional Entropy (Theoretical Φ): {phi:.4f}")
        print("    (Higher scores indicate greater likelihood of semantic drift)\n")
        print("-" * 60)
        print("Component Metrics:")
        print(f"  - Optimal number of topics (k): {metrics.get('optimal_k', 'N/A')}")
        
        print("\n--- Global (Aggregate-First) Divergence Metrics ---")
        print("    (Computed on the pooled distributions of all sentences)")
        print(f"  - Global JSD (Q||A): {metrics.get('global_jensen_shannon_divergence', 0.0):.4f}")
        print(f"  - Global KL(Prompt || Answer): {metrics.get('global_kl_divergence_prompt_answer', 0.0):.4f} bits")
        print(f"  - Global KL(Answer || Prompt): {metrics.get('global_kl_divergence_answer_prompt', 0.0):.4f} bits")
        print(f"  - Prompt Entropy: {metrics.get('global_prompt_entropy', 0.0):.4f} bits")
        print(f"  - Answer Entropy: {metrics.get('global_answer_entropy', 0.0):.4f} bits")
        print(f"  - Entropy Difference: {metrics.get('entropy_difference', 0.0):.4f}")
        print(f"  - Wasserstein Distance: {metrics.get('wasserstein_distance', 0.0):.4f}")

        # Display context metrics if available
        has_context = results.get('has_context', False)
        if has_context:
            print("\n--- Context Metrics (QCA Triplet Analysis) ---")
            print(f"  - Context Entropy: {metrics.get('global_context_entropy', 0.0):.4f} bits")
            print(f"  - Global JSD (C||A): {metrics.get('global_jsd_context_answer', 0.0):.4f}")
            print(f"  - Global JSD (Q||C): {metrics.get('global_jsd_prompt_context', 0.0):.4f}")
            print(f"  - Global KL(Context || Answer): {metrics.get('global_kl_divergence_context_answer', 0.0):.4f} bits")
            print(f"  - Global KL(Answer || Context): {metrics.get('global_kl_divergence_answer_context', 0.0):.4f} bits")
            print(f"  - Global KL(Prompt || Context): {metrics.get('global_kl_divergence_prompt_context', 0.0):.4f} bits")
            print(f"  - Global KL(Context || Prompt): {metrics.get('global_kl_divergence_context_prompt', 0.0):.4f} bits")

        print("\n--- Ensemble (Average-Later) Metrics ---")
        print("    (Computed locally for each paraphrase pair, then averaged)")
        print(f"  - Ensemble JSD: {metrics.get('ensemble_jensen_shannon_divergence', 0.0):.4f}  <-- Used in S_H Score")
        print(f"  - Ensemble KL(Prompt || Answer): {metrics.get('ensemble_kl_divergence_prompt_answer', 0.0):.4f} bits")
        print(f"  - Ensemble KL(Answer || Prompt): {metrics.get('ensemble_kl_divergence_answer_prompt', 0.0):.4f} bits")
        print(f"  - Ensemble MI (via H(Y)-H(Y|X)): {metrics.get('ensemble_mutual_information', 0.0):.4f} bits")
    
        print("\n--- Other Diagnostic Metrics ---")
        print(f"  - Averaged MI (from heatmap): {metrics.get('averaged_mutual_information', 0.0):.4f} bits")
        print(f"  - Adjusted MI (AMI): {metrics.get('adjusted_mutual_information', 0.0):.4f}")
    
        print("\n--- Semantic Entropy Baseline (Prior Art Method) ---")
        print(f"  - SE (Original Prompt Only): {metrics.get('semantic_entropy_original', 0.0):.4f} bits")
        print(f"  - Mean SE (Across Paraphrases): {metrics.get('mean_semantic_entropy', 0.0):.4f} bits")
        print("-" * 60)

    def _plot_averaged_cooccurrence_matrix(self, averaged_prob_matrix: np.ndarray, optimal_k: int, run_prefix: str):
        """
        (CORRECTED VERSION)
        Generates a heatmap of the pre-computed AVERAGED joint probability distribution.
        """
        if np.sum(averaged_prob_matrix) < 1e-9:
            print("Averaged probability matrix is empty or all zeros. Skipping heatmap.")
            return None
    
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            averaged_prob_matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=.5,
            cbar_kws={'label': 'Averaged Joint Probability P(X, Y)'}
        )
        plt.title("Averaged Topic Co-occurrence Distribution")
        plt.ylabel("Prompt Topic Index (X)")
        plt.xlabel("Answer Topic Index (Y)")
        
        file_path = f"{run_prefix}_averaged_cooccurrence_heatmap.png"
        plt.savefig(file_path)
        print(f"Averaged co-occurrence heatmap saved to: {file_path}")
        plt.close()
        
        return file_path

    def _calculate_semantic_entropy(self, answers_by_prompt, run_prefix, k_range):
        """
        Calculates the 'Semantic Entropy' (SE) of responses, as defined in prior work.
        This method clusters ONLY answer sentences.
    
        Returns a dictionary with two keys:
        - 'semantic_entropy': SE calculated on responses to the ORIGINAL prompt only.
        - 'mean_semantic_entropy': The mean of SE values calculated for each paraphrase.
        """
        print("Calculating Semantic Entropy (SE) baseline metrics...")
        
        # --- Variation 1: SE for the Original Prompt ---
        se_original = 0.0
        original_prompt_answers = answers_by_prompt[0]
        original_answer_sentences = [s for a in original_prompt_answers for s in self._split_into_sentences(a)]
        
        if original_answer_sentences:
            answer_embeddings = self._embed_texts(original_answer_sentences)
            # Estimate k and cluster for this specific set of answers
            se_k = self._estimate_optimal_clusters_elbow(
                answer_embeddings, f"{run_prefix}_se_original", k_range
            )
            if answer_embeddings.shape[0] > se_k:
                clustering = AgglomerativeClustering(n_clusters=se_k)
                labels = clustering.fit_predict(answer_embeddings)
                # Calculate entropy from the cluster label distribution
                counts = np.bincount(labels)
                probs = counts / np.sum(counts)
                se_original = entropy(probs, base=2)
    
        # --- Variation 2: Mean Semantic Entropy (MSE) across all paraphrases ---
        all_se_scores = []
        for i, answer_set in enumerate(answers_by_prompt):
            answer_sentences = [s for a in answer_set for s in self._split_into_sentences(a)]
            if answer_sentences:
                answer_embeddings = self._embed_texts(answer_sentences)
                se_k_i = self._estimate_optimal_clusters_elbow(
                    answer_embeddings, f"{run_prefix}_se_paraphrase_{i}", k_range
                )
                if answer_embeddings.shape[0] > se_k_i:
                    clustering = AgglomerativeClustering(n_clusters=se_k_i)
                    labels = clustering.fit_predict(answer_embeddings)
                    counts = np.bincount(labels)
                    probs = counts / np.sum(counts)
                    all_se_scores.append(entropy(probs, base=2))
    
        mean_se = np.mean(all_se_scores) if all_se_scores else 0.0
        
        return {
            "semantic_entropy_original": se_original,
            "mean_semantic_entropy": mean_se
        }        


