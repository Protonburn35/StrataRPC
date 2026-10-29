#include "ds4_session.hpp"
#include <iostream>

int main() {
    using namespace strata::ds4;
    namespace fs = std::filesystem;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) { if (!ok) { std::cerr << what << '\n'; ++failures; } };
    const fs::path root = fs::temp_directory_path() / ("ds4-session-test-" + session::new_id());
    fs::create_directories(root);
    const auto path = (root / "test.srpc").string();
    session::File file, restored;
    file.spec.identity_a = 123; file.spec.identity_b = 456;
    file.spec.n_layers = 40; file.spec.split = 20; file.spec.dim = 5120; file.spec.hc = 4;
    file.spec.vocab = 1000; file.spec.max_context = 1048576; file.spec.image_token = 17; file.spec.sessions = true;
    file.spec.wire = stage::LayerStageWireFormat::bf16;
    file.ref = {session::new_id(), 12345, 98765};
    file.images = {{10, 1024, 777}}; file.state.assign(20000, 0x5a);
    std::string err;
    uint64_t size = 0;
    check(session::write(path, file, size, err, true), "write failed");
    check(!session::write(path, file, size, err, true), "immutable worker snapshot overwritten");
    check(session::read(path, restored, size, err), "read failed");
    check(restored.state == file.state && restored.ref.id == file.ref.id && restored.ref.prefix == file.ref.prefix &&
          restored.images.size() == 1 && restored.images[0].hash == 777, "round trip differs");
    check(stage::same_spec(restored.spec, file.spec, err), "spec round trip differs");
    check(!session::read(path, restored, size, err, 1024), "restore ignored memory budget");
    // Replacing the coordinator file is supported, and a damaged payload must fail before any model load.
    file.state[0] ^= 1;
    check(session::write(path, file, size, err), "atomic replacement failed");
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(-9, std::ios::end); const char bad = 0; f.write(&bad, 1);
    }
    check(!session::read(path, restored, size, err), "damaged payload accepted");
    fs::resize_file(path, 40);
    check(!session::read(path, restored, size, err), "truncated metadata accepted");
    file.ref.id = "../bad";
    check(!session::write(path, file, size, err), "path-like snapshot ID accepted");
    fs::remove_all(root);
    return failures ? 1 : 0;
}
