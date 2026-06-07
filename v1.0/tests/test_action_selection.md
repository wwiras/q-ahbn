```bash
% python -m tests.test_action_selection

==============================
ACTION SELECTION TEST
==============================

State = ('M', 'L', 'L', 'L', 'L', 'L')

Q-values
ahbn_base                = 0.500
more_structured          = 1.200
more_gossip              = 3.500
duplicate_suppression    = 0.800
recovery_push            = 2.100
resource_conservative    = 0.300

Selected Action = more_gossip

RESULT
PASS: Policy selected highest-Q action.

```