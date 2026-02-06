#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int iJIT_IsProfilingActive(void) {
    return 0;
}

int iJIT_NotifyEvent(int eventType, void *EventSpecificData) {
    (void)eventType;
    (void)EventSpecificData;
    return 0;
}

int iJIT_GetNewMethodID(void) {
    return 0;
}

#ifdef __cplusplus
}
#endif
