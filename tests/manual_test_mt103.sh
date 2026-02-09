#!/bin/bash
# Manual test for MT103 query variations with LLM resolver

echo "======================================================================"
echo "Testing MT103 Query Variations with LLM Resolver"
echo "Expected count: 1855"
echo "======================================================================"
echo

variations=(
    "How many transactions with mt103 in December?"
    "Count transactions with MT103 in December"
    "How many MT103 transactions in December?"
    "Show count of transactions having mt103 in December"
)

for i in "${!variations[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Variation $((i+1)): ${variations[$i]}"
    echo "----------------------------------------------------------------------"
    
    haikugraph ask -q "${variations[$i]}" --use-llm-resolver --execute > /dev/null 2>&1
    
    # Extract count from result.json
    count=$(cat data/result.json | jq -r '.subquestion_results[0].preview_rows[0].count_transaction_id')
    echo "Result: $count"
    
    if [ "$count" == "1855" ]; then
        echo "✓ PASS"
    else
        echo "✗ FAIL (expected 1855, got $count)"
    fi
    
    echo
done

echo "======================================================================"
echo "Test complete"
echo "======================================================================"
