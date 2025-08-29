def compare_with_bert_benchmark(final_f1, final_accuracy, logger):
    bert_f1, bert_acc = 88.9, 84.8
    t5_f1 = final_f1 * 100
    t5_acc = (final_accuracy or 0.0) * 100
    
    print(f"Benchmark: BERT F1={bert_f1:.1f}% Acc={bert_acc:.1f}% | T5 F1={t5_f1:.1f}% Acc={t5_acc:.1f}%")
    logger.info(f"Benchmark - BERT: F1={bert_f1:.1f}% Acc={bert_acc:.1f}% | T5: F1={t5_f1:.1f}% Acc={t5_acc:.1f}%")