def compare_with_bert_benchmark(final_f1, final_accuracy, logger):
    """Compare our T5 model results with BERT benchmark on MRPC"""
    # BERT-base benchmark results on MRPC (from GLUE leaderboard/BERT paper)
    bert_f1 = 88.9
    bert_accuracy = 84.8
    
    # Convert percentages to same scale as our results
    bert_f1_scaled = bert_f1 / 100.0
    bert_accuracy_scaled = bert_accuracy / 100.0
    
    # Calculate differences
    f1_diff = final_f1 - bert_f1_scaled
    acc_diff = final_accuracy - bert_accuracy_scaled if final_accuracy else 0.0 - bert_accuracy_scaled
    
    # Create comparison message
    comparison_msg = f"""
Benchmark Comparison (MRPC Dataset):
  BERT-base (Benchmark):  F1={bert_f1:.1f}%  Accuracy={bert_accuracy:.1f}%
  T5-small (Our Model):   F1={final_f1*100:.1f}%  Accuracy={final_accuracy*100 if final_accuracy else 0.0:.1f}%
  Difference:             F1={f1_diff*100:+.1f}%  Accuracy={acc_diff*100:+.1f}%"""
    
    # Display in terminal
    print(comparison_msg)
    
    # Log to file
    logger.info("=== BENCHMARK COMPARISON ===")
    logger.info(f"BERT-base benchmark: F1={bert_f1:.1f}%, Accuracy={bert_accuracy:.1f}%")
    logger.info(f"T5-small our model: F1={final_f1*100:.1f}%, Accuracy={final_accuracy*100 if final_accuracy else 0.0:.1f}%")
    logger.info(f"Performance difference: F1={f1_diff*100:+.1f}%, Accuracy={acc_diff*100:+.1f}%")
    
    return comparison_msg