"""Export current editable Beta source and the verified v1.0.118 firmware.

The allowlist is independent of Git HEAD/history and never reads operator
outputs, private signing material or a device. Existing archives are not
overwritten. Factory seeds are copied from deployment, never from live data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from package_developer_snapshot import REPO, SOURCE_EXTS, payload_map, sanitize_config

sys.path.insert(0, str(REPO / "Python"))
from fbg_ota_protocol import (PACKAGE_HEADER, PACKAGE_HEADER_SIZE, PACKAGE_MAGIC,
                              PACKAGE_FORMAT_VERSION, OTA_LINK_ADDRESS,
                              OTA_TARGET_ID, crc32, validate_application_image)


_BUILT_FIRMWARE = REPO / "build/fw118_machine4_cmsis_20261012/JDSU_F205RE_v1.0.118_beta.fbgfw"
VERIFIED_FIRMWARE = (_BUILT_FIRMWARE if _BUILT_FIRMWARE.exists()
                     else REPO / "firmware/JDSU_F205RE_v1.0.118_beta.fbgfw")
FIRMWARE_VERSION = "1.0.118"
FIRMWARE_IMAGE_CRC32 = 0x1A951B61
FIRMWARE_IMAGE_SHA256 = "6f737b62ecd68819b895ae40abcd73589433074589a51dd37bf3757748791eec"
FIRMWARE_PACKAGE_SHA256 = "cb16345f99005105a2a861dc699934491a6735fa5e58e9e592f994b3ecddf255"
FIRMWARE_DESTINATION = "firmware/JDSU_F205RE_v1.0.118_beta.fbgfw"
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
                   ".release-signing", "beta_data", "build", "dist", "tmp", ".codex_tmp"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".exe", ".key", ".pem", ".pfx", ".p12",
                      ".axf", ".o", ".obj", ".log", ".bin", ".joblib", ".pickle", ".pkl"}
# These are reviewed static resources used by the existing development and
# freezer specs, not a scan of arbitrary calibration/measurement JSON files.
APPROVED_TOP_LEVEL_DATA = {
    "finger_sensor_layout.yaml", "t31_touch_calibration.json",
    "fullband_equal_power_operational_2001.json", "fullband_equal_power_operational_2001.csv",
    "fullband_dense_high_power_2001.json", "fullband_equal_power_operational_2001_transition_guards.json",
    "fullband_pdt_pdr_reference_2001.json", "laser_calibration_checkpoint.json",
    "laser_calibration_report.json", "mode_tables_from_fullband_2001.json",
    "rc_dynamic_model_validation.json", "rc_model_accuracy_63.json",
    "stress_high_power_135_final.json", "stress_realtime_benchmark_latest.json",
    "stress_shape_power_dynamic_v3.json", "stress_shape_power_final_v2.json",
    "wave_const.yaml", "stress_wave_const.yaml", "temporary_test_default_plan.json",
    "temporary_test_defaults.json", "machine_registry.json", "machine_profile.json",
    "usb_firmware_catalog.json",
}
APPROVED_DEPLOYMENT_DATA = {
    "deployment/universal/machine_profile.json", "deployment/machine1/machine_profile.json",
    "deployment/machine1/temporary_test_defaults.json", "deployment/machine2/machine_profile.json",
    "deployment/machine2/temporary_test_defaults.json",
}
FACTORY_SEED_PREFIXES = ("deployment/universal/client_seed/", "deployment/machine1/client_seed/",
                         "deployment/machine2/client_seed/")
REQUIRED_SOURCE = {
    "AGENTS.md", ".gitattributes", ".gitignore", "README_Beta源码与固件.md",
    "Python/app_version.py", "Python/launch_fbg_studio_beta.pyw",
    "Python/fbg_unified_app.py", "Python/beta_profile.py",
    "Python/offline_update_helper.py", "Python/offline_update_engine.py",
    "Python/dense_scan_transport.py", "Python/board_thermal_protocol.py",
    "Python/test_temporary_reference.py", "Python/tests/fixtures/forward_settle_analysis.json",
    "Python/tests/fixtures/forward_settle_capture.json",
    "Python/artifacts/ch1_measured_template_selection_v4_certified_20260905.json", "JDSU/AGENTS.md",
    "JDSU/Core/Inc/firmware_features.h", "JDSU/Core/Inc/firmware_version.h",
    "JDSU/MDK-ARM/JDSU/JDSU.sct", "JDSU/MDK-ARM/RTE/_JDSU/RTE_Components.h",
    "JDSU/MDK-ARM/JDSU.uvprojx", "JDSU/Bootloader/ota_bootloader.c",
    "JDSU/Drivers/CMSIS/Lib/ARM/arm_cortexM3l_math.lib",
    "deployment/build_beta_firmware.ps1", "deployment/pyinstaller/offline_update_helper.spec",
    FIRMWARE_DESTINATION,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def forbidden(relative: str) -> bool:
    path = PurePosixPath(relative)
    lowered = {part.lower() for part in path.parts}
    return (path.is_absolute() or ".." in path.parts or "\\" in relative
            or any(part in FORBIDDEN_PARTS or part.endswith("_private") for part in lowered)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or relative.startswith("Python/fbg_remote_config.") and not relative.endswith(".example")
            or relative.startswith("backups/")
            or relative == "JDSU/MDK-ARM/JDSU_beta_generated.uvprojx")


def literal_version_bytes(raw: bytes) -> tuple[str, str]:
    values = {}
    for statement in ast.parse(raw, filename="app_version.py").body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = statement.value.value
    return values["APP_VERSION"], values["APP_RELEASE_CHANNEL"]


def literal_source_version(path: Path) -> tuple[str, str]:
    return literal_version_bytes(path.read_bytes())


def source_beta_firmware_version(header: bytes) -> str:
    # Do not take the first PATCH macro: the unsupported polling branch is119.
    text = header.decode("utf-8-sig")
    branch = re.search(r"#elif\s+FBG_FEATURE_BETA_SINGLE_ADC\s*(.*?)#else", text, re.S)
    if not branch:
        raise ValueError("Missing supported single-ADC Beta firmware version branch")
    return ".".join(re.search(rf"APP_FIRMWARE_VERSION_{name}\s+(\d+)U", branch[1])[1]
                    for name in ("MAJOR", "MINOR", "PATCH"))


def verify_firmware_bytes(raw: bytes) -> dict:
    if len(raw) < PACKAGE_HEADER_SIZE:
        raise ValueError("Truncated Beta firmware package")
    (magic, fmt, header_size, target, address, version, size, _timestamp,
     reserved, body_crc, header_crc) = PACKAGE_HEADER.unpack_from(raw)
    image = raw[PACKAGE_HEADER_SIZE:]
    if (magic != PACKAGE_MAGIC or fmt != PACKAGE_FORMAT_VERSION or header_size != PACKAGE_HEADER_SIZE
            or target != OTA_TARGET_ID or address != OTA_LINK_ADDRESS or version != 0x10076
            or reserved != bytes(len(reserved)) or len(image) != size
            or header_crc != crc32(raw[:PACKAGE_HEADER_SIZE - 4])
            or body_crc != crc32(image)):
        raise ValueError("Invalid or unexpected Beta firmware package")
    validate_application_image(image)
    if (body_crc != FIRMWARE_IMAGE_CRC32 or digest(image) != FIRMWARE_IMAGE_SHA256
            or digest(raw) != FIRMWARE_PACKAGE_SHA256):
        raise ValueError("Firmware differs from the exact v1.0.118 package verified on machines3/4")
    return {"version": FIRMWARE_VERSION, "image_crc32": f"{body_crc:08X}",
            "image_bytes": len(image), "image_sha256": digest(image),
            "package_sha256": digest(raw), "link_address": f"0x{address:08X}"}


def select_payload(firmware: Path = VERIFIED_FIRMWARE) -> dict[str, Path]:
    raw_map = payload_map()
    resources = {"Python/" + name for name in APPROVED_TOP_LEVEL_DATA}
    resources.add("Python/artifacts/ch1_measured_template_selection_v4_certified_20260905.json")
    registry = json.loads((REPO / "Python/machine_registry.json").read_text(encoding="utf-8"))
    for config in registry["machines"].values():
        for name in ("source_path", "guard_path"):
            path = config.get("fullband", {}).get(name)
            if path:
                resources.add("Python/" + path)
    selected = {}
    for relative, source in raw_map.items():
        if forbidden(relative):
            continue
        suffix = source.suffix.lower()
        source_relative = source.relative_to(REPO).as_posix()
        factory = (source_relative in APPROVED_DEPLOYMENT_DATA
                   or source_relative.startswith(FACTORY_SEED_PREFIXES))
        config = relative.startswith("Python/machine_defaults/") and source.name == "temporary_test_defaults.json"
        fixture = relative.startswith("Python/tests/fixtures/")
        code = (suffix in SOURCE_EXTS or source.name.endswith(".example")
                or source.name in {".gitignore", ".mxproject", "LICENSE", "LICENSE.txt"})
        public_binary = relative.startswith("outputs/ota/") and suffix in {".fbgfw", ".hex"}
        if code or factory or config or fixture or relative in resources or suffix in {".stl", ".ico", ".dll"} or public_binary:
            selected[relative] = source
    selected["README.md"] = REPO / "README.md"
    selected["AGENTS.md"] = REPO / "AGENTS.md"
    selected[".gitattributes"] = REPO / ".gitattributes"
    selected.pop("README_开发调试.md", None)  # obsolete versioned developer template
    selected["README_Beta源码与固件.md"] = REPO / "deployment/beta_source_readme.md"
    seed = REPO / "deployment/universal/client_seed"
    for path in seed.rglob("*.json"):
        selected["Python/client_seed/" + path.relative_to(seed).as_posix()] = path
    legacy_seed = seed / "outputs/machines/machine_1/temporary_test"
    profile = json.loads((REPO / "Python/machine_profile.json").read_text(encoding="utf-8"))
    for item in profile["artifacts"]:
        relative = "Python/" + item["path"]
        if relative.startswith("Python/outputs/temporary_test/"):
            selected[relative] = legacy_seed / PurePosixPath(item["path"]).relative_to("outputs/temporary_test")
        if relative not in selected or digest(selected[relative].read_bytes()) != item["sha256"]:
            raise ValueError("Pinned calibration resource mismatch: " + relative)
    for path in (REPO / "JDSU/Drivers").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".lib", ".a"}:
            selected[path.relative_to(REPO).as_posix()] = path
    catalog = json.loads((REPO / "Python/usb_firmware_catalog.json").read_text(encoding="utf-8"))
    for item in catalog["machines"].values():
        filename = item["filename"]
        if not re.fullmatch(r"JDSU_F205RE_v\d+\.\d+\.\d+\.fbgfw", filename):
            raise ValueError("Unsafe compatibility firmware filename")
        source = REPO / "outputs/ota" / filename
        if digest(source.read_bytes()) != item["sha256"]:
            raise ValueError("Approved compatibility firmware hash mismatch")
        selected["Python/firmware/usb/" + item["filename"]] = source
    firmware = firmware.resolve()
    verify_firmware_bytes(firmware.read_bytes())
    selected[FIRMWARE_DESTINATION] = firmware
    missing = REQUIRED_SOURCE - selected.keys()
    if missing:
        raise ValueError("Missing required editable source/resources: " + ", ".join(sorted(missing)))
    return selected


def safe_payload(relative: str, source: Path) -> tuple[bytes, list[str]]:
    if forbidden(relative):
        raise ValueError("Forbidden archive path: " + relative)
    raw = source.read_bytes()
    fields = []
    if source.suffix.lower() == ".json":
        safe = sanitize_config(json.loads(raw.decode("utf-8-sig")), fields)
        if fields:
            raw = (json.dumps(safe, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if source.suffix.lower() in {".py", ".pyw"}:
        ast.parse(raw, filename=relative)
    if len(raw) >= 100 * 1024 * 1024:
        raise ValueError("Unexpected oversized source resource: " + relative)
    return raw, fields


def verify_archive(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as zip_file:
        if zip_file.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        manifests = [name for name in zip_file.namelist() if name.endswith("/MANIFEST.sha256.json")]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one source manifest")
        manifest = json.loads(zip_file.read(manifests[0]))
        prefix = manifests[0].removesuffix("MANIFEST.sha256.json")
        paths = [entry["path"] for entry in manifest["files"]]
        names = [prefix + path for path in paths] + manifests
        if len(set(names)) != len(names) or set(names) != set(zip_file.namelist()):
            raise ValueError("Duplicate, unlisted or missing archive member")
        if REQUIRED_SOURCE - set(paths):
            raise ValueError("Required source missing from archive")
        for entry in manifest["files"]:
            relative = entry["path"]
            if forbidden(relative):
                raise ValueError("Forbidden archive member: " + relative)
            data = zip_file.read(prefix + relative)
            if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
                raise ValueError("Manifest hash mismatch: " + relative)
            suffix = PurePosixPath(relative).suffix.lower()
            if suffix in {".py", ".pyw"}:
                ast.parse(data, filename=relative)
            if suffix == ".json":
                fields = []
                sanitize_config(json.loads(data.decode("utf-8-sig")), fields)
                if fields:
                    raise ValueError("Nonempty sensitive JSON fields: " + relative)
        checked = verify_firmware_bytes(zip_file.read(prefix + FIRMWARE_DESTINATION))
        if checked != manifest["firmware"]:
            raise ValueError("Firmware manifest metadata mismatch")
        version, channel = literal_version_bytes(zip_file.read(prefix + "Python/app_version.py"))
        if version != manifest["client_version"] or channel != manifest["release_channel"] or channel != "beta-offline":
            raise ValueError("Client source version/channel differs from manifest")
        if source_beta_firmware_version(zip_file.read(prefix + "JDSU/Core/Inc/firmware_version.h")) != FIRMWARE_VERSION:
            raise ValueError("Archive is not the supported single-ADC Beta source")
        profile = json.loads(zip_file.read(prefix + "Python/machine_profile.json"))
        for item in profile["artifacts"]:
            data = zip_file.read(prefix + "Python/" + item["path"])
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError("Archive calibration pin mismatch: " + item["path"])
        catalog = json.loads(zip_file.read(prefix + "Python/usb_firmware_catalog.json"))
        for item in catalog["machines"].values():
            data = zip_file.read(prefix + "Python/firmware/usb/" + item["filename"])
            if digest(data) != item["sha256"]:
                raise ValueError("Archive USB compatibility firmware pin mismatch")
        return manifest


def git_baseline() -> str | None:
    # Extracted developer bundles are intentionally not Git repositories.
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_archive(client_version: str, output_directory: Path, firmware: Path = VERIFIED_FIRMWARE) -> dict:
    actual, channel = literal_source_version(REPO / "Python/app_version.py")
    if client_version != actual or not re.fullmatch(r"\d+\.\d+\.\d+", actual) or channel != "beta-offline":
        raise ValueError("Requested client version/channel differs from actual Beta source")
    if source_beta_firmware_version((REPO / "JDSU/Core/Inc/firmware_version.h").read_bytes()) != FIRMWARE_VERSION:
        raise ValueError("Current single-ADC Beta source is not v1.0.118")
    selected = select_payload(firmware)
    output = output_directory.resolve()
    output.relative_to(REPO / "release")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_name = f"OptiSense_Beta_Source_v{actual}_FW{FIRMWARE_VERSION}_{stamp}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (root_name + ".zip")
    partial = archive.with_suffix(".zip.partial")
    entries = []
    source_digests = {}
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        for relative, source in sorted(selected.items()):
            original = source.read_bytes()
            checksum = digest(original)
            source_digests[source] = checksum
            data, fields = safe_payload(relative, source)
            if source.read_bytes() != original:
                raise ValueError("Worktree changed while reading source: " + relative)
            zip_file.writestr(root_name + "/" + relative, data)
            entry = {"path": relative, "bytes": len(data), "sha256": digest(data),
                     "source_path": source.relative_to(REPO).as_posix(), "source_sha256": checksum}
            if fields:
                entry["sanitized_fields"] = fields
            entries.append(entry)
        manifest = {"schema": "fbg_beta_editable_source_bundle_v1", "client_version": actual,
                    "release_channel": channel, "firmware": verify_firmware_bytes(firmware.read_bytes()),
                    "snapshot": "current worktree files, including uncommitted/untracked source; not git archive",
                    "base_git_commit": git_baseline(), "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "python": "CPython3.12 x64", "files": entries,
                    "exclusions": ["signing private keys", "live operator outputs and beta_data", "private full Flash",
                                   "Git history", "venv/build caches", "customer EXE/installers", "large model binaries"]}
        zip_file.writestr(root_name + "/MANIFEST.sha256.json",
                          json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    verify_archive(partial)
    for source, checksum in source_digests.items():
        if digest(source.read_bytes()) != checksum:
            raise ValueError("Worktree changed while packaging: " + source.relative_to(REPO).as_posix())
    partial.rename(archive)
    checksum = digest(archive.read_bytes())
    archive.with_suffix(".zip.sha256").write_text(checksum + "  " + archive.name + "\n", encoding="utf-8")
    return {"zip": str(archive), "bytes": archive.stat().st_size, "sha256": checksum,
            "files": len(entries), "client_version": actual, "firmware": manifest["firmware"],
            "verified": True, "sanitized_files": sum("sanitized_fields" in item for item in entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-version")
    parser.add_argument("--firmware", type=Path, default=VERIFIED_FIRMWARE)
    parser.add_argument("--output-directory", type=Path, default=REPO / "release/beta_source")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_archive(args.verify)
        report = {"verified": True, "files": len(manifest["files"]), "client_version": manifest["client_version"],
                  "firmware": manifest["firmware"]}
    else:
        if not args.client_version:
            parser.error("--client-version is required for a new source snapshot")
        report = create_archive(args.client_version, args.output_directory, args.firmware)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
