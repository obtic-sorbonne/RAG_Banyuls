class EvaluationManager:
    def __init__(self, config_path="evaluation_config.yaml"):
        self.config = self.load_config(config_path)
        self.tracker = ExperimentTracker()
        self.retrieval_metrics = {
            "precision_recall": PrecisionRecallCalculator(),
            "ndcg": NDCGCalculator(),
            "mrr": MRRCalculator(),
            "coverage": CoverageCalculator()
        }
        self.generation_metrics = {
            "factual_consistency": FactualConsistency(),
            "fluency": FluencyEvaluator(),
            "relevance": RelevanceEvaluator(),
            "hallucination": HallucinationDetector()
        }
        self.end_to_end_metrics = {
            "human_eval": HumanEvaluationFramework(),
            "user_satisfaction": UserSatisfactionTracker(),
            "task_completion": TaskCompletionEvaluator()
        }
    
    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def evaluate_retrieval(self, queries: dict, run_id: str = None):
        """Evaluate retrieval performance for a set of queries"""
        if run_id:
            self.tracker.start_run("retrieval_evaluation", self.config['retrieval'])
        
        results = {}
        for query_id, data in queries.items():
            query_results = {}
            for metric_name, calculator in self.retrieval_metrics.items():
                if hasattr(calculator, 'calculate'):
                    query_results[metric_name] = calculator.calculate(
                        data['retrieved'], 
                        data['relevant'],
                        k=self.config['retrieval'].get('k', 5)
                    )
                elif hasattr(calculator, 'calculate_batch'):
                    query_results[metric_name] = calculator.calculate_batch(
                        {query_id: data}
                    )
            
            results[query_id] = query_results
            
            if run_id:
                self.tracker.log_metrics(query_results, step=query_id)
        
        if run_id:
            self.tracker.end_run()
        
        return results
    
    def evaluate_generation(self, responses: list, run_id: str = None):
        """Evaluate generation quality for a set of responses"""
        if run_id:
            self.tracker.start_run("generation_evaluation", self.config['generation'])
        
        results = {}
        for i, response_data in enumerate(responses):
            response_results = {}
            for metric_name, calculator in self.generation_metrics.items():
                if metric_name == "factual_consistency":
                    response_results[metric_name] = calculator.calculate(
                        response_data['response'],
                        response_data['context']
                    )
                elif metric_name == "relevance":
                    response_results[metric_name] = calculator.calculate(
                        response_data['response'],
                        response_data['query']
                    )
                else:
                    response_results[metric_name] = calculator.calculate(
                        response_data['response']
                    )
            
            results[f"response_{i}"] = response_results
            
            if run_id:
                self.tracker.log_metrics(response_results, step=i)
        
        if run_id:
            self.tracker.end_run()
        
        return results
    
    def run_end_to_end_evaluation(self, eval_data: list, run_id: str = None):
        """Conduct comprehensive end-to-end evaluation"""
        if run_id:
            self.tracker.start_run("end_to_end_evaluation", self.config['end_to_end'])
        
        results = {}
        
        # Human evaluation
        human_metrics = self.end_to_end_metrics['human_eval'].calculate_aggregates()
        results['human_evaluation'] = human_metrics
        
        # User satisfaction
        satisfaction = self.end_to_end_metrics['user_satisfaction'].calculate_satisfaction_score()
        results['user_satisfaction'] = satisfaction
        
        # Task completion
        task_results = []
        for task in eval_data:
            if 'task_id' in task and 'success_criteria' in task:
                self.end_to_end_metrics['task_completion'].define_task(
                    task['task_id'], task['success_criteria']
                )
                completed = self.end_to_end_metrics['task_completion'].evaluate_completion(
                    task['task_id'], task['query'], task['response']
                )
                task_results.append(completed)
        
        completion_rate = self.end_to_end_metrics['task_completion'].batch_evaluate(task_results)
        results['task_completion_rate'] = completion_rate
        
        if run_id:
            self.tracker.log_metrics(results)
            self.tracker.end_run()
        
        return results
    
    def analyze_failures(self, failure_data: dict):
        """Conduct failure analysis across components"""
        analysis = {}
        
        # Retrieval failures
        analysis['retrieval'] = FailureAnalyzer().analyze_retrieval_failures()
        
        # Generation hallucinations
        analysis['generation'] = FailureAnalyzer().analyze_generation_hallucinations()
        
        # End-to-end issues
        feedback_themes = self.end_to_end_metrics['user_satisfaction'].analyze_feedback()
        analysis['user_feedback'] = feedback_themes
        
        return analysis
