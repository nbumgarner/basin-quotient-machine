/* checkpoint_dump.h — Write int8 hidden state to /tmp for cosine comparison.
 * Include after bqsm_server.c's variable declarations. Usage:
 *   CP_DUMP("embed", x, dm);
 */
static void cp_dump(const char *label, const int8_t *data, int len) {
    char fn[128];
    snprintf(fn, sizeof(fn), "/tmp/bqsm_%s.bin", label);
    FILE *f = fopen(fn, "wb");
    if (f) { fwrite(data, 1, (size_t)len, f); fclose(f); }
}
#define CP_DUMP(label, data, len) cp_dump(label, data, len)
