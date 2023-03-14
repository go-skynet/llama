#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *llama_allocate_state();

int llama_bootstrap(const char *model_path, void *state_pr);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                                                        float top_p, float temp, float repeat_penalty);
void llama_free_params(void* params_ptr);

int llama_predict(void* params_ptr, void* state_pr);

#ifdef __cplusplus
}
#endif
