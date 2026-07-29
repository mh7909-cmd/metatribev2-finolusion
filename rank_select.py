def rank_and_select(scored_variants: list[dict]) -> tuple[dict, list[dict]]:
    """
    Ranks the list of scored variants in descending order, prints a
    visual leaderboard, and selects the highest-scoring variant.
    """
    # Sort descending by score
    ranked_list = sorted(scored_variants, key=lambda x: x["score"], reverse=True)
    
    selected_variant = ranked_list[0]
    
    # Calculate baseline (lowest score in the batch)
    min_score = min(v["score"] for v in scored_variants)
    
    print("\n" + "=" * 90)
    print("                    TRIBE v2 ENGAGEMENT RANKING LEADERBOARD                    ")
    print("=" * 90)
    print(f"{'Rank':<5} | {'Engagement Score':<18} | {'Rel. Delta':<10} | {'Brain Region Activations':<35} | {'Phrasing'}")
    print("-" * 90)
    
    for idx, item in enumerate(ranked_list, 1):
        score = item["score"]
        text = item["text"]
        act = item["activations"]
        act_str = f"PFC:{act['PFC']:.2f} Amy:{act['Amygdala']:.2f} Temp:{act['Temporal']:.2f} NAcc:{act['NAcc']:.2f}"
        
        # Calculate relative delta compared to the lowest variant in the batch
        if min_score > 0 and score != min_score:
            rel_delta = ((score - min_score) / min_score) * 100
            rel_delta_str = f"+{rel_delta:.2f}%"
        else:
            rel_delta_str = "Baseline"
            
        # Highlight the winner
        prefix = "★" if idx == 1 else " "
        
        # Truncate text if too long for preview
        preview_text = text if len(text) < 30 else text[:27] + "..."
        
        print(f"{prefix} #{idx:<2} | {score:<18.4f} | {rel_delta_str:<10} | {act_str:<35} | \"{preview_text}\"")
        
    print("=" * 90)
    print(f"Winner Selected: \"{selected_variant['text']}\"\n")
    
    return selected_variant, ranked_list

if __name__ == "__main__":
    # Test ranking locally
    test_variants = [
        {"text": "Option A is short.", "score": 0.456, "activations": {"PFC": 0.5, "Amygdala": 0.3, "Temporal": 0.6, "NAcc": 0.4}},
        {"text": "Option B is highly engaging and exciting!", "score": 0.789, "activations": {"PFC": 0.7, "Amygdala": 0.9, "Temporal": 0.7, "NAcc": 0.85}},
        {"text": "Option C is direct.", "score": 0.521, "activations": {"PFC": 0.6, "Amygdala": 0.4, "Temporal": 0.5, "NAcc": 0.5}}
    ]
    rank_and_select(test_variants)
