```bash
% python -m tests.test_learning_validation

==============================
Q-AHBN LEARNING VALIDATION
==============================
Episodes            : 500
Q Updates           : 486
States Learned      : 233
Mean Reward         : 3.181
Recent Reward       : 3.183
Final Epsilon       : 0.030

ACTION DISTRIBUTION
more_structured            113
ahbn_base                   88
resource_conservative       86
duplicate_suppression       82
more_gossip                 76
recovery_push               55

CSV Written: outputs/csv/learning_trace_20260616_183205.csv
```