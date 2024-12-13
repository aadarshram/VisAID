This is Ad's work in progress to add Long term memory to companion

Issues:

0. although loading existing vectorstore doesnt seem to remember. idk why.
1. Using inmemory docstore so no memory across scripts.
2. once recall memory is updated with one memory. it doesnt query for a new memory query instead just says sry dont have in memory. prolly u need to get rid of recall memories in state- implies every time it will query database. nothing saved in context. find better method,

3. prompt tuning needed to align outputs of different nodes appropriately in all cases.
