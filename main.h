// num.h
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *llama_allocate_state();

bool llama_bootstrap(const char *model_path, void *state_pr);

void* llama_allocate_params(const char *input, int threads, int tokens);
void llama_free_params(void* params_ptr);

int llama_predict(void* params_ptr, void* state_pr);

#ifdef __cplusplus
}
#endif
