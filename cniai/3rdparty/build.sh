#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(readlink -f "$0")
THIRD_PARTY_DIR=$(dirname "$SCRIPT_PATH")

CNIAI_INSTALL_PATH=${THIRD_PARTY_DIR}/../install


THIRD_PARTY_BUILD_DIR=${THIRD_PARTY_DIR}/build
if [ ! -d "${THIRD_PARTY_BUILD_DIR}" ]; then
    mkdir -p "${THIRD_PARTY_BUILD_DIR}"
fi


install_gflags() {
    GFLAGS_PATH=${THIRD_PARTY_DIR}/gflags
    rm -rf "${THIRD_PARTY_BUILD_DIR}"/gflags && mkdir -p "${THIRD_PARTY_BUILD_DIR}"/gflags && cd "${THIRD_PARTY_BUILD_DIR}"/gflags
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="${CNIAI_INSTALL_PATH}" "${GFLAGS_PATH}"
    make -j $(($(nproc) + 1)) && make install
    return 0
}


install_spdlog() {
    SPDLOG_PATH=${THIRD_PARTY_DIR}/spdlog
    rm -rf "${THIRD_PARTY_BUILD_DIR}"/spdlog && mkdir -p "${THIRD_PARTY_BUILD_DIR}"/spdlog && cd "${THIRD_PARTY_BUILD_DIR}"/spdlog
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="${CNIAI_INSTALL_PATH}" "${SPDLOG_PATH}"
    make -j $(($(nproc) + 1)) && make install
    return 0
}


install() {
  install_gflags || return $?

  install_spdlog || return $?

  return 0
}


if [ -z "$1" ]; then
    install
    exit 0
fi


for arg in "$@"
do
    if [[ "gflags" == "${arg}" ]]; then
        install_gflags
    fi

    if [[ "spdlog" == "${arg}" ]]; then
        install_spdlog
    fi
done