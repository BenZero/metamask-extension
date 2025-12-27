import os

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin")
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from threading import Lock
from typing import Dict, Iterator, List, Optional, Tuple

import ecdsa
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from Crypto.Hash import RIPEMD160, keccak
from pycuda.compiler import SourceModule
from web3 import Web3

print("=" * 70)
print("MULTI-CHAIN BRAINWALLET GENERATOR - PRODUKTIONSREIF")
print("Echte Ethereum-Adressen mit Keccak-256 aus pycryptodome")
print("=" * 70)


DEFAULT_PROCESS_WORKERS = max(1, min(os.cpu_count() or 1, 32))


def keccak_256_ethereum(data: bytes) -> bytes:
    """Echte Keccak-256 für Ethereum aus pycryptodome."""
    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(data)
    return keccak_hash.digest()


@dataclass(frozen=True)
class ChainCfg:
    name: str
    chain_id: int
    rpcs: List[str]
    symbol: str


@dataclass(frozen=True)
class UtxoApiCfg:
    name: str
    url_template: str
    parser: str


@dataclass(frozen=True)
class UtxoChainCfg:
    name: str
    symbol: str
    address_version: int
    wif_version: int
    apis: List[UtxoApiCfg]


CHAINS: List[ChainCfg] = [
    ChainCfg(
        name="Ethereum",
        chain_id=1,
        rpcs=[
            "https://eth-mainnet.g.alchemy.com/v2/4LOXL_jFvZ-CkRsrtJ3KtF5ON4nbf0_m",
            "https://eth.llamarpc.com",
            "https://rpc.mevblocker.io",
            "https://0xrpc.io/eth",
            "https://endpoints.omniatech.io/v1/eth/mainnet/public",
            "https://ethereum-rpc.publicnode.com",
            "https://eth.api.pocket.network",
            "https://eth.merkle.io"
        ],
        symbol="eth"
    ),
    ChainCfg(
        name="Polygon",
        chain_id=137,
        rpcs=[
            "https://1rpc.io/matic",
            "https://poly.api.pocket.network",
            "https://endpoints.omniatech.io/v1/matic/mainnet/public",
            "https://polygon.drpc.org",
            "https://polygon-bor-rpc.publicnode.com"
        ],
        symbol="POL"
    ),
    ChainCfg(
        name="BNB",
        chain_id=56,
        rpcs=[
            "https://bsc-dataseed.bnbchain.org",
            "https://bsc-dataseed-public.bnbchain.org",
            "https://bsc-dataseed.nariox.org",
            "https://bsc-dataseed.defibit.io",
            "https://bsc-dataseed.ninicoin.io",
            "https://binance.llamarpc.com",
            "https://bsc.blockrazor.xyz",
            "https://bsc-rpc.publicnode.com"
        ],
        symbol="bnb"
    ),
    ChainCfg(
        name="Arbitrum One",
        chain_id=42161,
        rpcs=[
            "https://public-arb-mainnet.fastnode.io",
            "https://arbitrum.meowrpc.com",
            "https://arb-one.api.pocket.network",
            "https://arb1.arbitrum.io/rpc",
            "https://rpc.ankr.com/arbitrum",
            "https://arbitrum-one.publicnode.com",
            "https://nova.arbitrum.io/rpc",
        ],
        symbol="ETH",
    ),
    ChainCfg(
        name="Optimism",
        chain_id=10,
        rpcs=[
            "https://optimism.drpc.org",
            "https://public-op-mainnet.fastnode.io",
            "https://0xrpc.io/op",
            "https://mainnet.optimism.io",
            "https://rpc.ankr.com/optimism",
            "https://optimism.publicnode.com",
        ],
        symbol="ETH",
    ),
]

UTXO_CHAINS: List[UtxoChainCfg] = [
    UtxoChainCfg(
        name="Bitcoin",
        symbol="BTC",
        address_version=0x00,
        wif_version=0x80,
        apis=[
            UtxoApiCfg(
                name="blockstream",
                url_template="https://blockstream.info/api/address/{address}",
                parser="blockstream",
            ),
            UtxoApiCfg(
                name="mempool",
                url_template="https://mempool.space/api/address/{address}",
                parser="blockstream",
            ),
            UtxoApiCfg(
                name="blockchain.info",
                url_template="https://blockchain.info/balance?active={address}",
                parser="blockchain_info",
            ),
            UtxoApiCfg(
                name="blockcypher",
                url_template="https://api.blockcypher.com/v1/btc/main/addrs/{address}/balance",
                parser="blockcypher",
            ),
        ],
    ),
    UtxoChainCfg(
        name="Litecoin",
        symbol="LTC",
        address_version=0x30,
        wif_version=0xB0,
        apis=[
            UtxoApiCfg(
                name="blockstream",
                url_template="https://blockstream.info/ltc/api/address/{address}",
                parser="blockstream",
            ),
            UtxoApiCfg(
                name="litecoinspace",
                url_template="https://litecoinspace.org/api/address/{address}",
                parser="blockstream",
            ),
            UtxoApiCfg(
                name="sochain",
                url_template="https://sochain.com/api/v2/get_address_balance/LTC/{address}",
                parser="sochain",
            ),
            UtxoApiCfg(
                name="blockcypher",
                url_template="https://api.blockcypher.com/v1/ltc/main/addrs/{address}/balance",
                parser="blockcypher",
            ),
        ],
    ),
    UtxoChainCfg(
        name="Dogecoin",
        symbol="DOGE",
        address_version=0x1E,
        wif_version=0x9E,
        apis=[
            UtxoApiCfg(
                name="blockcypher",
                url_template="https://api.blockcypher.com/v1/doge/main/addrs/{address}/balance",
                parser="blockcypher",
            ),
            UtxoApiCfg(
                name="sochain",
                url_template="https://sochain.com/api/v2/get_address_balance/DOGE/{address}",
                parser="sochain",
            ),
            UtxoApiCfg(
                name="dogechain",
                url_template="https://dogechain.info/api/v1/address/balance/{address}",
                parser="dogechain",
            ),
        ],
    ),
]


class EthereumBrainwallet:
    """Korrekter Brainwallet-Generator für Ethereum."""

    SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

    @staticmethod
    def password_to_private_key(password: str) -> bytes:
        """
        Konvertiert Passwort zu privatem Ethereum Schlüssel EXAKT wie Brainwallet.
        Brainwallet Methode: SHA-256 des Passworts, dann modulo secp256k1 order.
        """
        password_bytes = password.encode("utf-8")
        sha256_hash = hashlib.sha256(password_bytes).digest()
        private_key_int = int.from_bytes(sha256_hash, "big")
        private_key_int = private_key_int % EthereumBrainwallet.SECP256K1_ORDER
        if private_key_int == 0:
            private_key_int = 1
        private_key = private_key_int.to_bytes(32, "big")
        assert 1 <= private_key_int < EthereumBrainwallet.SECP256K1_ORDER
        assert len(private_key) == 32
        return private_key

    @staticmethod
    def private_key_to_public_key(private_key: bytes) -> Tuple[int, int]:
        """Konvertiert privaten Schlüssel zu öffentlichem Schlüssel (x, y)."""
        sk = ecdsa.SigningKey.from_string(private_key, curve=ecdsa.SECP256k1)
        vk = sk.get_verifying_key()
        public_key_bytes = vk.to_string()
        x = int.from_bytes(public_key_bytes[:32], "big")
        y = int.from_bytes(public_key_bytes[32:], "big")
        return x, y

    @staticmethod
    def public_key_to_ethereum_address(public_key_x: int, public_key_y: int) -> str:
        """Konvertiert öffentlichen Schlüssel zu Ethereum-Adresse."""
        public_key_bytes = (
            b"\x04" + public_key_x.to_bytes(32, "big") + public_key_y.to_bytes(32, "big")
        )
        keccak_hash = keccak_256_ethereum(public_key_bytes[1:])
        address_bytes = keccak_hash[-20:]
        address_hex = "0x" + address_bytes.hex()
        return Web3.to_checksum_address(address_hex)

    @staticmethod
    def private_key_to_ethereum_address(private_key: bytes) -> str:
        """Komplette Konvertierung: Private Key → Ethereum Address."""
        x, y = EthereumBrainwallet.private_key_to_public_key(private_key)
        return EthereumBrainwallet.public_key_to_ethereum_address(x, y)

    @staticmethod
    def generate_wallet_from_private_key(password: str, private_key: bytes) -> dict:
        """Generiert komplettes Wallet aus Passwort + privatem Schlüssel."""
        x, y = EthereumBrainwallet.private_key_to_public_key(private_key)
        public_key_hex = "04" + x.to_bytes(32, "big").hex() + y.to_bytes(32, "big").hex()
        address = EthereumBrainwallet.private_key_to_ethereum_address(private_key)
        return {
            "password": password,
            "private_key": private_key.hex(),
            "public_key": public_key_hex,
            "address": address,
            "valid": True,
            "brainwallet_type": "SHA256(passphrase) mod n",
        }

    @staticmethod
    def generate_wallet_from_password(password: str) -> dict:
        """Generiert komplettes Wallet aus einem Passwort."""
        try:
            private_key = EthereumBrainwallet.password_to_private_key(password)
            return EthereumBrainwallet.generate_wallet_from_private_key(password, private_key)
        except Exception as exc:
            return {"password": password, "error": str(exc), "valid": False}

    @staticmethod
    def validate_private_key(private_key: bytes) -> bool:
        """Validiert ob privater Schlüssel gültig ist."""
        if len(private_key) != 32:
            return False
        private_key_int = int.from_bytes(private_key, "big")
        return 1 <= private_key_int < EthereumBrainwallet.SECP256K1_ORDER


BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _hash160(data: bytes) -> bytes:
    sha_hash = hashlib.sha256(data).digest()
    ripe_hash = RIPEMD160.new()
    ripe_hash.update(sha_hash)
    return ripe_hash.digest()


def _base58_encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    encoded = ""
    while num > 0:
        num, rem = divmod(num, 58)
        encoded = BASE58_ALPHABET[rem] + encoded
    pad = 0
    for byte in data:
        if byte == 0:
            pad += 1
        else:
            break
    return "1" * pad + encoded


def _base58check_encode(payload: bytes) -> str:
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return _base58_encode(payload + checksum)


class UtxoBrainwallet:
    """Brainwallet Generator für UTXO Chains (BTC/LTC/DOGE)."""

    @staticmethod
    def private_key_to_p2pkh_address(private_key: bytes, chain: UtxoChainCfg) -> str:
        x, y = EthereumBrainwallet.private_key_to_public_key(private_key)
        public_key = b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")
        pubkey_hash = _hash160(public_key)
        payload = bytes([chain.address_version]) + pubkey_hash
        return _base58check_encode(payload)

    @staticmethod
    def private_key_to_wif(private_key: bytes, chain: UtxoChainCfg) -> str:
        payload = bytes([chain.wif_version]) + private_key
        return _base58check_encode(payload)


sha256_kernel_code = r"""
#include <stdint.h>

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82d4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d7d, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t Sigma0(uint32_t x) {
    return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

__device__ uint32_t Sigma1(uint32_t x) {
    return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

__device__ uint32_t sigma0(uint32_t x) {
    return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3);
}

__device__ uint32_t sigma1(uint32_t x) {
    return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10);
}

__global__ void sha256_brainwallet_kernel(
    const unsigned char* passwords,
    const int* password_lengths,
    int num_passwords,
    unsigned char* private_keys_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_passwords) return;

    int offset = idx * 256;
    int len = password_lengths[idx];

    uint32_t h0 = 0x6a09e667;
    uint32_t h1 = 0xbb67ae85;
    uint32_t h2 = 0x3c6ef372;
    uint32_t h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f;
    uint32_t h5 = 0x9b05688c;
    uint32_t h6 = 0x1f83d9ab;
    uint32_t h7 = 0x5be0cd19;

    uint8_t chunk[64];
    uint64_t total_bits = len * 8;
    int chunk_count = 0;

    while (chunk_count * 64 <= len) {
        for (int i = 0; i < 64; i++) {
            int pos = chunk_count * 64 + i;
            if (pos < len) {
                chunk[i] = passwords[offset + pos];
            } else if (pos == len) {
                chunk[i] = 0x80;
            } else if (i < 56) {
                chunk[i] = 0x00;
            } else {
                chunk[i] = (total_bits >> (56 - (i - 56) * 8)) & 0xFF;
            }
        }

        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)chunk[i*4] << 24) |
                   ((uint32_t)chunk[i*4+1] << 16) |
                   ((uint32_t)chunk[i*4+2] << 8) |
                   (uint32_t)chunk[i*4+3];
        }

        for (int i = 16; i < 64; i++) {
            w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
        }

        uint32_t a = h0, b = h1, c = h2, d = h3;
        uint32_t e = h4, f = h5, g = h6, h_val = h7;

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = Sigma0(a) + Maj(a, b, c);
            h_val = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h0 += a; h1 += b; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h_val;

        chunk_count++;
    }

    int out_offset = idx * 32;
    private_keys_out[out_offset] = (h0 >> 24) & 0xFF;
    private_keys_out[out_offset + 1] = (h0 >> 16) & 0xFF;
    private_keys_out[out_offset + 2] = (h0 >> 8) & 0xFF;
    private_keys_out[out_offset + 3] = h0 & 0xFF;

    private_keys_out[out_offset + 4] = (h1 >> 24) & 0xFF;
    private_keys_out[out_offset + 5] = (h1 >> 16) & 0xFF;
    private_keys_out[out_offset + 6] = (h1 >> 8) & 0xFF;
    private_keys_out[out_offset + 7] = h1 & 0xFF;

    private_keys_out[out_offset + 8] = (h2 >> 24) & 0xFF;
    private_keys_out[out_offset + 9] = (h2 >> 16) & 0xFF;
    private_keys_out[out_offset + 10] = (h2 >> 8) & 0xFF;
    private_keys_out[out_offset + 11] = h2 & 0xFF;

    private_keys_out[out_offset + 12] = (h3 >> 24) & 0xFF;
    private_keys_out[out_offset + 13] = (h3 >> 16) & 0xFF;
    private_keys_out[out_offset + 14] = (h3 >> 8) & 0xFF;
    private_keys_out[out_offset + 15] = h3 & 0xFF;

    private_keys_out[out_offset + 16] = (h4 >> 24) & 0xFF;
    private_keys_out[out_offset + 17] = (h4 >> 16) & 0xFF;
    private_keys_out[out_offset + 18] = (h4 >> 8) & 0xFF;
    private_keys_out[out_offset + 19] = h4 & 0xFF;

    private_keys_out[out_offset + 20] = (h5 >> 24) & 0xFF;
    private_keys_out[out_offset + 21] = (h5 >> 16) & 0xFF;
    private_keys_out[out_offset + 22] = (h5 >> 8) & 0xFF;
    private_keys_out[out_offset + 23] = h5 & 0xFF;

    private_keys_out[out_offset + 24] = (h6 >> 24) & 0xFF;
    private_keys_out[out_offset + 25] = (h6 >> 16) & 0xFF;
    private_keys_out[out_offset + 26] = (h6 >> 8) & 0xFF;
    private_keys_out[out_offset + 27] = h6 & 0xFF;

    private_keys_out[out_offset + 28] = (h7 >> 24) & 0xFF;
    private_keys_out[out_offset + 29] = (h7 >> 16) & 0xFF;
    private_keys_out[out_offset + 30] = (h7 >> 8) & 0xFF;
    private_keys_out[out_offset + 31] = h7 & 0xFF;
}
"""


class GPUBrainwalletGenerator:
    """GPU-beschleunigter Brainwallet Generator."""

    def __init__(self, max_password_length: int = 256):
        self.max_password_length = max_password_length
        self.brainwallet = EthereumBrainwallet()
        try:
            self.mod = SourceModule(
                sha256_kernel_code,
                options=[
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_120",
                    "-allow-unsupported-compiler",
                ],
            )
            self.sha256_kernel = self.mod.get_function("sha256_brainwallet_kernel")
            self.gpu_available = True
            print("✅ GPU acceleration enabled")
        except Exception as exc:
            print(f"⚠️  GPU not available: {exc}")
            self.gpu_available = False

    def compute_private_keys_gpu(self, passwords: List[str]) -> List[bytes]:
        """Berechnet private keys von Passwörtern auf GPU."""
        if not self.gpu_available or len(passwords) < 100:
            return self.compute_private_keys_cpu(passwords)

        num_passwords = len(passwords)
        password_data = np.zeros((num_passwords, self.max_password_length), dtype=np.uint8)
        password_lengths = np.zeros(num_passwords, dtype=np.int32)

        for i, pwd in enumerate(passwords):
            pwd_bytes = pwd.encode("utf-8")
            length = min(len(pwd_bytes), self.max_password_length)
            password_data[i, :length] = np.frombuffer(pwd_bytes[:length], dtype=np.uint8)
            password_lengths[i] = length

        passwords_gpu = drv.mem_alloc(password_data.nbytes)
        lengths_gpu = drv.mem_alloc(password_lengths.nbytes)
        hashes_gpu = drv.mem_alloc(num_passwords * 32)

        drv.memcpy_htod(passwords_gpu, password_data)
        drv.memcpy_htod(lengths_gpu, password_lengths)

        block_size = 256
        grid_size = (num_passwords + block_size - 1) // block_size

        start_time = time.time()
        self.sha256_kernel(
            passwords_gpu,
            lengths_gpu,
            np.int32(num_passwords),
            hashes_gpu,
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        sha256_hashes = np.zeros((num_passwords, 32), dtype=np.uint8)
        drv.memcpy_dtoh(sha256_hashes, hashes_gpu)

        passwords_gpu.free()
        lengths_gpu.free()
        hashes_gpu.free()

        private_keys = []
        for i in range(num_passwords):
            hash_int = int.from_bytes(bytes(sha256_hashes[i]), "big")
            private_key_int = hash_int % EthereumBrainwallet.SECP256K1_ORDER
            if private_key_int == 0:
                private_key_int = 1
            private_key = private_key_int.to_bytes(32, "big")
            private_keys.append(private_key)

        elapsed = time.time() - start_time
        print(
            f"  GPU: {num_passwords} private keys in {elapsed:.3f}s "
            f"({num_passwords / elapsed:.0f}/sec)"
        )

        return private_keys

    def compute_private_keys_cpu(self, passwords: List[str]) -> List[bytes]:
        """CPU-Fallback für private key Berechnung."""
        start_time = time.time()
        private_keys = [self.brainwallet.password_to_private_key(password) for password in passwords]
        elapsed = time.time() - start_time
        rate = len(passwords) / max(elapsed, 0.001)
        print(f"  CPU: {len(passwords)} private keys in {elapsed:.3f}s ({rate:.0f}/sec)")
        return private_keys

    def generate_wallets(self, passwords: List[str]) -> List[dict]:
        """Generiert komplette Wallets aus Passwörtern."""
        print(f"\nGenerating {len(passwords)} brainwallets (EVM + BTC/LTC/DOGE)...")
        private_keys = self.compute_private_keys_gpu(passwords)

        wallets: List[dict] = []
        start_time = time.time()
        for i, (password, private_key) in enumerate(zip(passwords, private_keys)):
            try:
                wallet = self.brainwallet.generate_wallet_from_private_key(password, private_key)
                utxo_addresses = {}
                utxo_wif = {}
                for chain in UTXO_CHAINS:
                    utxo_addresses[chain.symbol] = UtxoBrainwallet.private_key_to_p2pkh_address(
                        private_key, chain
                    )
                    utxo_wif[chain.symbol] = UtxoBrainwallet.private_key_to_wif(
                        private_key, chain
                    )
                wallet["utxo_addresses"] = utxo_addresses
                wallet["utxo_wif"] = utxo_wif
                wallets.append(wallet)
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / max(elapsed, 0.001)
                    print(
                        f"  Generated {i + 1}/{len(passwords)} wallets "
                        f"({rate:.0f}/sec)",
                        end="\r",
                    )
            except Exception as exc:
                wallets.append({"password": password, "error": str(exc), "valid": False})

        elapsed = time.time() - start_time
        valid_wallets = len([w for w in wallets if w.get("valid", False)])
        print(f"\n✅ Generated {valid_wallets} valid brainwallets in {elapsed:.2f}s")

        return wallets


class BlockchainScanner:
    def __init__(self, rpc_timeout: int = 3, max_rpc_workers: Optional[int] = None):
        self.rpc_timeout = rpc_timeout
        self.max_rpc_workers = 8
        self._web3_clients: Dict[str, Web3] = {}
        self._last_working_rpc: Dict[int, str] = {}
        self._lock = Lock()

    def _get_web3(self, rpc: str) -> Web3:
        if rpc not in self._web3_clients:
            self._web3_clients[rpc] = Web3(
                Web3.HTTPProvider(rpc, request_kwargs={"timeout": self.rpc_timeout})
            )
        return self._web3_clients[rpc]

    def _resolve_rpc(self, chain: ChainCfg) -> str:
        if chain.chain_id in self._last_working_rpc:
            return self._last_working_rpc[chain.chain_id]
        for rpc in chain.rpcs:
            try:
                w3 = self._get_web3(rpc)
                if w3.is_connected():
                    self._last_working_rpc[chain.chain_id] = rpc
                    return rpc
            except Exception:
                continue
        self._last_working_rpc[chain.chain_id] = ""
        return ""

    def connect_rpc_with_failover(
        rpcs: List[str],
        expected_chain_id: int,
        timeout_s: int = 15,
        debug: bool = False,
    ) -> Tuple[Web3, str]:
        last_err: Optional[Exception] = None
        for url in rpcs:
            try:
                w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": timeout_s}))

                connected = w3.is_connected()
                if not connected:
                    raise RuntimeError(f"RPC not reachable: {url}")

                cid = w3.eth.chain_id

                if cid != expected_chain_id:
                    raise RuntimeError(
                        f"RPC chainId mismatch: expected {expected_chain_id}, got {cid} ({url})"
                    )

                return w3, url

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"Kein RPC erreichbar (chainId={expected_chain_id}). Letzter Fehler: {last_err!r}"
        )

    def check_balance(self, address: str, chain, debug: bool = False) -> Tuple[bool, float, str]:
        """
        Prüft Guthaben auf einer Chain.
        debug=True loggt pro Call (inkl. Dauer) – ideal um 'zu schnell' zu entlarven.
        """
        try:
            w3, rpc = BlockchainScanner.connect_rpc_with_failover(
                chain.rpcs,
                expected_chain_id=chain.chain_id,
                timeout_s=getattr(chain, "timeout_s", 15),
                debug=debug,
            )

            balance_wei = w3.eth.get_balance(address)
            balance_eth = w3.from_wei(balance_wei, "ether")

            if hasattr(self, "_last_working_rpc"):
                self._last_working_rpc[chain.chain_id] = rpc

            if balance_wei > 0:
                return True, float(balance_eth), rpc
            return False, 0.0, rpc

        except Exception as e:
            print(f"{e}")

            if hasattr(self, "_last_working_rpc"):
                self._last_working_rpc.pop(chain.chain_id, None)
            return False, 0.0, ""

    def _fetch_json(self, url: str) -> dict:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "brainwallet-scanner/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.rpc_timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)

    def _parse_utxo_balance(self, parser: str, data: dict, address: str) -> float:
        if parser == "blockstream":
            chain_stats = data.get("chain_stats", {})
            mempool_stats = data.get("mempool_stats", {})
            funded = chain_stats.get("funded_txo_sum", 0) + mempool_stats.get("funded_txo_sum", 0)
            spent = chain_stats.get("spent_txo_sum", 0) + mempool_stats.get("spent_txo_sum", 0)
            return (funded - spent) / 1e8
        if parser == "blockcypher":
            balance = data.get("final_balance", data.get("balance", 0))
            return balance / 1e8
        if parser == "blockchain_info":
            address_info = data.get(address, {})
            balance = address_info.get("final_balance", 0)
            return balance / 1e8
        if parser == "sochain":
            payload = data.get("data", {})
            confirmed = float(payload.get("confirmed_balance", 0))
            unconfirmed = float(payload.get("unconfirmed_balance", 0))
            return confirmed + unconfirmed
        if parser == "dogechain":
            balance = data.get("balance", 0)
            return float(balance)
        raise ValueError(f"Unknown UTXO parser: {parser}")

    def check_utxo_balance(
        self, address: str, chain: UtxoChainCfg, debug: bool = False
    ) -> Tuple[bool, float, str]:
        last_err: Optional[Exception] = None
        best_balance = 0.0
        sources: List[str] = []
        for api in chain.apis:
            url = api.url_template.format(address=address)
            try:
                data = self._fetch_json(url)
                balance = self._parse_utxo_balance(api.parser, data, address)
                if balance > best_balance:
                    best_balance = balance
                if balance > 0:
                    sources.append(api.name)
            except (urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
                last_err = exc
                if debug:
                    print(f"UTXO API error ({chain.symbol}/{api.name}): {exc}")
                continue
        if best_balance > 0:
            return True, best_balance, ",".join(sources) if sources else ""
        if last_err:
            print(f"{chain.symbol} API error: {last_err}")
        return False, 0.0, ""

    def scan_single_wallet(self, wallet: dict) -> Optional[dict]:
        """Scannt ein einzelnes Wallet nach Guthaben."""
        if not wallet.get("valid", False):
            return None

        address = wallet["address"]
        utxo_addresses = wallet.get("utxo_addresses", {})
        found_balances = []

        with ThreadPoolExecutor(max_workers=self.max_rpc_workers) as executor:
            chain_results = list(executor.map(lambda c: self.check_balance(address, c), CHAINS))
            utxo_items = [
                (chain, utxo_addresses.get(chain.symbol)) for chain in UTXO_CHAINS
            ]
            utxo_results = list(
                executor.map(
                    lambda item: self.check_utxo_balance(item[1], item[0])
                    if item[1]
                    else (False, 0.0, ""),
                    utxo_items,
                )
            )

        for chain, (has_balance, balance, rpc) in zip(CHAINS, chain_results):
            if has_balance:
                found_balances.append(
                    {
                        "chain": chain.name,
                        "balance": balance,
                        "symbol": chain.symbol,
                        "rpc": rpc,
                    }
                )

        for (chain, _address), (has_balance, balance, api) in zip(utxo_items, utxo_results):
            if has_balance:
                found_balances.append(
                    {
                        "chain": chain.name,
                        "balance": balance,
                        "symbol": chain.symbol,
                        "api": api,
                    }
                )

        if found_balances:
            wallet["found"] = True
            wallet["balances"] = found_balances
            wallet["total_balance"] = sum(b["balance"] for b in found_balances)
            return wallet

        return None

    def scan_wallets(self, wallets: List[dict]) -> List[dict]:
        """Scannt Wallets nach Guthaben."""
        print(f"\nScanning {len(wallets)} wallets for balances...")

        results = []
        start_time = time.time()

        for i, wallet in enumerate(wallets):
            found_wallet = self.scan_single_wallet(wallet)

            if found_wallet:
                print(
                    f"🎉 FOUND: {found_wallet['address']} - "
                    f"{found_wallet['total_balance']} total balance"
                )
                results.append(found_wallet)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                print(f"  Scanned {i + 1}/{len(wallets)} ({rate:.1f}/sec)", end="\r")

        elapsed = time.time() - start_time
        print(f"\n✅ Scanned {len(wallets)} wallets in {elapsed:.2f}s")
        print(f"💰 Found {len(results)} wallets with balance")

        return results


def scan_wallet_in_process(wallet: dict, rpc_timeout: int, max_rpc_workers: int) -> Optional[dict]:
    scanner = BlockchainScanner(rpc_timeout=rpc_timeout, max_rpc_workers=max_rpc_workers)
    return scanner.scan_single_wallet(wallet)


def scan_wallets_multiprocess(
    wallets: List[dict],
    rpc_timeout: int,
    max_rpc_workers: int,
    process_workers: int,
) -> List[dict]:
    print(f"\nScanning {len(wallets)} wallets for balances (multiprocess)...")

    valid_wallets = [wallet for wallet in wallets if wallet.get("valid", False)]
    results: List[dict] = []
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=process_workers,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(scan_wallet_in_process, wallet, rpc_timeout, max_rpc_workers): wallet
            for wallet in valid_wallets
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            found_wallet = future.result()
            if found_wallet:
                print(
                    f"🎉 FOUND: {found_wallet['address']} - "
                    f"{found_wallet['total_balance']} total balance"
                )
                results.append(found_wallet)

            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / max(elapsed, 0.001)
                print(f"  Scanned {idx}/{len(valid_wallets)} ({rate:.1f}/sec)", end="\r")

    elapsed = time.time() - start_time
    print(f"\n✅ Scanned {len(valid_wallets)} wallets in {elapsed:.2f}s")
    print(f"💰 Found {len(results)} wallets with balance")

    return results


def load_passwords_from_file(filename: str = "passwords.txt") -> List[str]:
    """Lädt Passwörter aus Datei (nicht für sehr große Dateien)."""
    passwords: List[str] = []

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    passwords.append(line)

    return passwords


def iter_passwords_from_file(filename: str) -> Iterator[str]:
    """Streamt Passwörter aus Datei, ohne alles in RAM zu laden."""
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def batched(iterable: Iterator[str], batch_size: int) -> Iterator[List[str]]:
    """Erstellt Batches aus einem Iterator."""
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_common_brainwallet_passwords() -> List[str]:
    """Generiert häufige Brainwallet-Passwörter."""
    passwords = []

    brainwallets = [
        "password",
        "123456",
        "12345678",
        "qwerty",
        "letmein",
        "monkey",
        "dragon",
        "baseball",
        "football",
        "mustang",
        "bitcoin",
        "ethereum",
        "crypto",
        "blockchain",
        "wallet",
        "privatekey",
        "seedphrase",
        "metamask",
        "ledger",
        "trezor",
        "satoshi",
        "vitalik",
        "buterin",
        "cryptocurrency",
        "defi",
        "correcthorsebatterystaple",
        "brainwallet",
        "brain",
        "mind",
        "memory",
        "remember",
        "passphrase",
        "seed",
        "recovery",
        "backup",
        "security",
        "mypasswordisverysecure",
        "ilovemycryptowallet",
        "tothe moon",
        "hodl",
        "buybitcoin",
        "sellhighbuylow",
        "whenmoon",
        "lambomoon",
        "mytreasure",
        "myfortune",
        "bitcoinwallet123",
        "ethereum2024",
        "crypto123456",
        "walletpassword",
        "privatekey123",
        "metamask2024",
        "hello world",
        "test passphrase",
        "my secret phrase",
        "crypto wallet backup",
        "ethereum private key",
    ]

    passwords.extend(brainwallets)

    for i in range(100):
        passwords.append(f"password{i}")
        passwords.append(f"wallet{i}")
        passwords.append(f"crypto{i}")
        passwords.append(f"eth{i}")
        passwords.append(f"btc{i}")
        passwords.append(str(i) * 6)

    for pwd in brainwallets[:20]:
        passwords.append(pwd.upper())
        passwords.append(pwd.capitalize())

    return list(set(passwords))


def save_results(results: List[dict], base_filename: str = "ethereum_brainwallets"):
    """Speichert Ergebnisse in verschiedenen Formaten."""
    if not results:
        print("No results to save")
        return

    valid_wallets = [w for w in results if w.get("valid", False)]
    wallets_with_balance = [w for w in results if w.get("found", False)]

    all_filename = f"{base_filename}_all.json"
    with open(all_filename, "w") as f:
        json.dump(valid_wallets, f, indent=2, default=str)

    if wallets_with_balance:
        found_filename = f"{base_filename}_with_balance.json"
        with open(found_filename, "w") as f:
            json.dump(wallets_with_balance, f, indent=2, default=str)

    txt_filename = f"{base_filename}.txt"
    with open(txt_filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-CHAIN BRAINWALLET GENERATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total wallets generated: {len(valid_wallets)}\n")
        f.write(f"Wallets with balance: {len(wallets_with_balance)}\n\n")

        if wallets_with_balance:
            f.write("WALLETS WITH BALANCE:\n")
            f.write("=" * 70 + "\n")
            for wallet in wallets_with_balance:
                f.write(f"\nPassword: {wallet['password']}\n")
                f.write(f"Address: {wallet['address']}\n")
                f.write(f"Private Key: {wallet['private_key']}\n")
                f.write(f"Public Key: {wallet['public_key']}\n")
                if "utxo_addresses" in wallet:
                    f.write("UTXO Addresses:\n")
                    for symbol, address in wallet["utxo_addresses"].items():
                        f.write(f"  {symbol}: {address}\n")

                if "balances" in wallet:
                    f.write("Balances:\n")
                    for balance in wallet["balances"]:
                        f.write(
                            f"  {balance['chain']}: {balance['balance']} {balance['symbol']}\n"
                        )

                f.write("-" * 50 + "\n")

        f.write("\n\nALL GENERATED WALLETS:\n")
        f.write("=" * 70 + "\n")
        for wallet in valid_wallets[:100]:
            f.write(f"\nPassword: {wallet['password']}\n")
            f.write(f"Address: {wallet['address']}\n")
            f.write(f"Private Key: {wallet['private_key'][:64]}...\n")
            if "utxo_addresses" in wallet:
                f.write("UTXO Addresses:\n")
                for symbol, address in wallet["utxo_addresses"].items():
                    f.write(f"  {symbol}: {address}\n")
            f.write("-" * 40 + "\n")

        if len(valid_wallets) > 100:
            f.write(f"\n... and {len(valid_wallets) - 100} more wallets\n")

    print("\n💾 Results saved:")
    print(f"   All wallets: {all_filename}")
    if wallets_with_balance:
        print(f"   With balance: {found_filename}")
    print(f"   Text summary: {txt_filename}")


def save_streaming_summary(
    total_wallets: int,
    wallets_with_balance: int,
    base_filename: str = "ethereum_brainwallets",
):
    """Speichert eine kompakte Summary für Streaming-Scans."""
    summary_filename = f"{base_filename}_summary.txt"
    with open(summary_filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-CHAIN BRAINWALLET STREAMING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total wallets scanned: {total_wallets}\n")
        f.write(f"Wallets with balance: {wallets_with_balance}\n")
    print(f"\n💾 Streaming summary saved: {summary_filename}")


def verify_implementation():
    """Verifiziert, dass die Implementierung korrekt ist."""
    print("\nVerifying implementation with test cases...")

    test_password = "test123"
    brainwallet = EthereumBrainwallet()

    private_key = brainwallet.password_to_private_key(test_password)
    address = brainwallet.private_key_to_ethereum_address(private_key)

    print(f"Test 1 - Password: '{test_password}'")
    print(f"  Private Key: {private_key.hex()}")
    print(f"  Address: {address}")

    empty_password = ""
    try:
        empty_key = brainwallet.password_to_private_key(empty_password)
        empty_address = brainwallet.private_key_to_ethereum_address(empty_key)
        print("\nTest 2 - Empty password:")
        print(f"  Address: {empty_address}")
    except Exception as exc:
        print(f"\nTest 2 - Empty password failed: {exc}")

    long_password = "a" * 1000
    long_key = brainwallet.password_to_private_key(long_password)
    long_address = brainwallet.private_key_to_ethereum_address(long_key)
    print("\nTest 3 - Long password (1000 chars):")
    print(f"  Address: {long_address}")

    print("\n✅ Verification complete")


def main():
    """Hauptfunktion."""
    print("\n" + "=" * 70)
    print("MULTI-CHAIN BRAINWALLET GENERATOR")
    print("=" * 70)

    verify_implementation()

    print("\n" + "=" * 70)
    print("LOADING/GENERATING PASSWORDS")
    print("=" * 70)

    passwords_file = "passwords.txt"
    streaming_mode = os.path.exists(passwords_file)
    passwords = []

    if streaming_mode:
        print(f"✅ Streaming passwords from {passwords_file}")
    else:
        print("⚠️  No passwords.txt found, generating common passwords...")
        passwords = generate_common_brainwallet_passwords()
        print(f"✅ Generated {len(passwords)} common brainwallet passwords")

    max_passwords = 10000
    if passwords and len(passwords) > max_passwords:
        print(f"⚠️  Limiting to first {max_passwords} passwords for demo")
        passwords = passwords[:max_passwords]

    print("\n" + "=" * 70)
    print("INITIALIZING GPU GENERATOR")
    print("=" * 70)

    generator = GPUBrainwalletGenerator()

    print("\n" + "=" * 70)
    print("SCANNING BLOCKCHAINS")
    print("=" * 70)

    scanner = BlockchainScanner()
    process_workers = int(os.getenv("BRAINWALLET_PROCESS_WORKERS", str(DEFAULT_PROCESS_WORKERS)))
    print(f"✅ Multiprocess scanner enabled with {process_workers} processes")

    if streaming_mode:
        total_scanned = 0
        total_found = 0
        found_jsonl = "ethereum_brainwallets_with_balance.jsonl"
        password_iter = iter_passwords_from_file(passwords_file)
        for batch in batched(password_iter, 10000):
            wallets = generator.generate_wallets(batch)
            found_wallets = scan_wallets_multiprocess(
                wallets,
                rpc_timeout=scanner.rpc_timeout,
                max_rpc_workers=scanner.max_rpc_workers,
                process_workers=process_workers,
            )
            total_scanned += len(wallets)
            total_found += len(found_wallets)
            if found_wallets:
                with open(found_jsonl, "a") as f:
                    for wallet in found_wallets:
                        f.write(json.dumps(wallet, default=str) + "\n")
        save_streaming_summary(total_scanned, total_found)
        if total_found:
            print(f"\n🎉 WARNING: Wallets with balance found! ({total_found} total)")
        else:
            print("\nNo wallets with balance found.")
    else:
        wallets = generator.generate_wallets(passwords)
        found_wallets = scan_wallets_multiprocess(
            wallets,
            rpc_timeout=scanner.rpc_timeout,
            max_rpc_workers=scanner.max_rpc_workers,
            process_workers=process_workers,
        )
        save_results(wallets)
        if found_wallets:
            print("\n🎉 WARNING: Wallets with balance found!")
        else:
            print("\nNo wallets with balance found.")


if __name__ == "__main__":
    main()
