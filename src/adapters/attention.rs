//! # Attention State Serialization
//!
//! Format for storing retrievable attention states, not just text.
//!
//! ## The Key Insight
//!
//! Traditional RAG stores text and re-embeds on retrieval.
//! HAT stores **attention states** that can be directly injected into LLM context.
//!
//! ## What Gets Stored
//!
//! For each memory chunk:
//! - **Text**: Original tokens/content
//! - **Embedding**: Vector for retrieval routing
//! - **KV Cache**: Compressed key-value states (optional, model-specific)
//! - **Metadata**: Timestamp, role, session context
//!
//! ## Format Design
//!
//! ```text
//! AttentionState
//! ├── id: Id (16 bytes)
//! ├── timestamp_ms: u64
//! ├── role: Role (user/assistant/system)
//! ├── text: String (original content)
//! ├── embedding: Vec<f32> (for HAT routing)
//! ├── kv_cache: Option<CompressedKV> (model-specific)
//! └── metadata: HashMap<String, String>
//! ```

use crate::core::Id;

/// Role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System prompt
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// Tool/function call
    Tool,
    /// Retrieved context (from RAG or previous HAT retrieval)
    Context,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Context => "context",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" | "function" => Some(Role::Tool),
            "context" | "retrieved" => Some(Role::Context),
            _ => None,
        }
    }

    fn to_byte(&self) -> u8 {
        match self {
            Role::System => 0,
            Role::User => 1,
            Role::Assistant => 2,
            Role::Tool => 3,
            Role::Context => 4,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Role::System),
            1 => Some(Role::User),
            2 => Some(Role::Assistant),
            3 => Some(Role::Tool),
            4 => Some(Role::Context),
            _ => None,
        }
    }
}

/// Compressed KV cache for a specific model architecture
///
/// This is model-specific. Different models have different:
/// - Number of layers
/// - Number of heads
/// - Head dimensions
/// - Quantization formats
#[derive(Debug, Clone)]
pub struct CompressedKV {
    /// Model identifier (e.g., "llama-3-8b", "mistral-7b")
    pub model_id: String,

    /// Number of layers
    pub num_layers: u32,

    /// Number of attention heads
    pub num_heads: u32,

    /// Dimension per head
    pub head_dim: u32,

    /// Sequence length this KV cache covers
    pub seq_len: u32,

    /// Quantization format (e.g., "fp16", "int8", "int4")
    pub quantization: String,

    /// Compressed KV data
    /// Format: [layer][head][seq][key/value][head_dim]
    /// Actual layout depends on quantization
    pub data: Vec<u8>,
}

impl CompressedKV {
    /// Estimate memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Create a placeholder (for models that don't support KV export)
    pub fn placeholder(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            num_layers: 0,
            num_heads: 0,
            head_dim: 0,
            seq_len: 0,
            quantization: "none".to_string(),
            data: vec![],
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Model ID (length-prefixed string)
        let model_bytes = self.model_id.as_bytes();
        bytes.extend_from_slice(&(model_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(model_bytes);

        // Architecture params
        bytes.extend_from_slice(&self.num_layers.to_le_bytes());
        bytes.extend_from_slice(&self.num_heads.to_le_bytes());
        bytes.extend_from_slice(&self.head_dim.to_le_bytes());
        bytes.extend_from_slice(&self.seq_len.to_le_bytes());

        // Quantization (length-prefixed string)
        let quant_bytes = self.quantization.as_bytes();
        bytes.extend_from_slice(&(quant_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(quant_bytes);

        // Data (length-prefixed)
        bytes.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.data);

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        let mut offset = 0;

        // Model ID
        if data.len() < offset + 4 {
            return None;
        }
        let model_len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        offset += 4;

        if data.len() < offset + model_len {
            return None;
        }
        let model_id = String::from_utf8(data[offset..offset + model_len].to_vec()).ok()?;
        offset += model_len;

        // Architecture params
        if data.len() < offset + 16 {
            return None;
        }
        let num_layers = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let num_heads = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let head_dim = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let seq_len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;

        // Quantization
        if data.len() < offset + 4 {
            return None;
        }
        let quant_len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        offset += 4;

        if data.len() < offset + quant_len {
            return None;
        }
        let quantization = String::from_utf8(data[offset..offset + quant_len].to_vec()).ok()?;
        offset += quant_len;

        // Data
        if data.len() < offset + 8 {
            return None;
        }
        let data_len = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?) as usize;
        offset += 8;

        if data.len() < offset + data_len {
            return None;
        }
        let kv_data = data[offset..offset + data_len].to_vec();
        offset += data_len;

        Some((
            Self {
                model_id,
                num_layers,
                num_heads,
                head_dim,
                seq_len,
                quantization,
                data: kv_data,
            },
            offset,
        ))
    }
}

/// A complete attention state for a memory chunk
#[derive(Debug, Clone)]
pub struct AttentionState {
    /// Unique identifier
    pub id: Id,

    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,

    /// Role in conversation
    pub role: Role,

    /// Original text content
    pub text: String,

    /// Embedding vector (for HAT retrieval routing)
    pub embedding: Vec<f32>,

    /// Optional compressed KV cache (model-specific)
    pub kv_cache: Option<CompressedKV>,

    /// Additional metadata (flexible key-value pairs)
    pub metadata: std::collections::HashMap<String, String>,
}

impl AttentionState {
    /// Create a new attention state (without KV cache)
    pub fn new(role: Role, text: String, embedding: Vec<f32>) -> Self {
        Self {
            id: Id::now(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            role,
            text,
            embedding,
            kv_cache: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create with KV cache
    pub fn with_kv_cache(mut self, kv: CompressedKV) -> Self {
        self.kv_cache = Some(kv);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Estimate total size in bytes
    pub fn size_bytes(&self) -> usize {
        16 + // id
        8 +  // timestamp
        1 +  // role
        self.text.len() +
        self.embedding.len() * 4 +
        self.kv_cache.as_ref().map(|kv| kv.size_bytes()).unwrap_or(0) +
        self.metadata.iter().map(|(k, v)| k.len() + v.len() + 8).sum::<usize>()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic + version
        bytes.extend_from_slice(b"ATTN");
        bytes.extend_from_slice(&1u32.to_le_bytes());

        // ID
        bytes.extend_from_slice(self.id.as_bytes());

        // Timestamp
        bytes.extend_from_slice(&self.timestamp_ms.to_le_bytes());

        // Role
        bytes.push(self.role.to_byte());

        // Text (length-prefixed)
        let text_bytes = self.text.as_bytes();
        bytes.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(text_bytes);

        // Embedding (length-prefixed)
        bytes.extend_from_slice(&(self.embedding.len() as u32).to_le_bytes());
        for &v in &self.embedding {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // KV cache (present flag + data)
        if let Some(ref kv) = self.kv_cache {
            bytes.push(1);
            let kv_bytes = kv.to_bytes();
            bytes.extend_from_slice(&(kv_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&kv_bytes);
        } else {
            bytes.push(0);
        }

        // Metadata (count + entries)
        bytes.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        for (key, value) in &self.metadata {
            let key_bytes = key.as_bytes();
            let value_bytes = value.as_bytes();
            bytes.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(key_bytes);
            bytes.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value_bytes);
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, AttentionError> {
        let mut offset = 0;

        // Magic
        if data.len() < 8 {
            return Err(AttentionError::InvalidFormat("Too short".into()));
        }
        if &data[0..4] != b"ATTN" {
            return Err(AttentionError::InvalidMagic);
        }
        offset += 4;

        // Version
        let version = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        if version != 1 {
            return Err(AttentionError::UnsupportedVersion(version));
        }
        offset += 4;

        // ID
        if data.len() < offset + 16 {
            return Err(AttentionError::InvalidFormat("Missing ID".into()));
        }
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&data[offset..offset + 16]);
        let id = Id::from_bytes(id_bytes);
        offset += 16;

        // Timestamp
        if data.len() < offset + 8 {
            return Err(AttentionError::InvalidFormat("Missing timestamp".into()));
        }
        let timestamp_ms = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Role
        if data.len() < offset + 1 {
            return Err(AttentionError::InvalidFormat("Missing role".into()));
        }
        let role = Role::from_byte(data[offset])
            .ok_or_else(|| AttentionError::InvalidFormat("Invalid role".into()))?;
        offset += 1;

        // Text
        if data.len() < offset + 4 {
            return Err(AttentionError::InvalidFormat("Missing text length".into()));
        }
        let text_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if data.len() < offset + text_len {
            return Err(AttentionError::InvalidFormat("Text truncated".into()));
        }
        let text = String::from_utf8(data[offset..offset + text_len].to_vec())
            .map_err(|_| AttentionError::InvalidFormat("Invalid UTF-8 in text".into()))?;
        offset += text_len;

        // Embedding
        if data.len() < offset + 4 {
            return Err(AttentionError::InvalidFormat("Missing embedding length".into()));
        }
        let emb_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if data.len() < offset + emb_len * 4 {
            return Err(AttentionError::InvalidFormat("Embedding truncated".into()));
        }
        let mut embedding = Vec::with_capacity(emb_len);
        for _ in 0..emb_len {
            embedding.push(f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()));
            offset += 4;
        }

        // KV cache
        if data.len() < offset + 1 {
            return Err(AttentionError::InvalidFormat("Missing KV flag".into()));
        }
        let has_kv = data[offset] != 0;
        offset += 1;

        let kv_cache = if has_kv {
            if data.len() < offset + 8 {
                return Err(AttentionError::InvalidFormat("Missing KV length".into()));
            }
            let kv_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;

            if data.len() < offset + kv_len {
                return Err(AttentionError::InvalidFormat("KV data truncated".into()));
            }
            let (kv, _) = CompressedKV::from_bytes(&data[offset..offset + kv_len])
                .ok_or_else(|| AttentionError::InvalidFormat("Invalid KV cache".into()))?;
            offset += kv_len;
            Some(kv)
        } else {
            None
        };

        // Metadata
        if data.len() < offset + 4 {
            return Err(AttentionError::InvalidFormat("Missing metadata count".into()));
        }
        let meta_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let mut metadata = std::collections::HashMap::new();
        for _ in 0..meta_count {
            // Key
            if data.len() < offset + 4 {
                return Err(AttentionError::InvalidFormat("Missing key length".into()));
            }
            let key_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if data.len() < offset + key_len {
                return Err(AttentionError::InvalidFormat("Key truncated".into()));
            }
            let key = String::from_utf8(data[offset..offset + key_len].to_vec())
                .map_err(|_| AttentionError::InvalidFormat("Invalid UTF-8 in key".into()))?;
            offset += key_len;

            // Value
            if data.len() < offset + 4 {
                return Err(AttentionError::InvalidFormat("Missing value length".into()));
            }
            let value_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if data.len() < offset + value_len {
                return Err(AttentionError::InvalidFormat("Value truncated".into()));
            }
            let value = String::from_utf8(data[offset..offset + value_len].to_vec())
                .map_err(|_| AttentionError::InvalidFormat("Invalid UTF-8 in value".into()))?;
            offset += value_len;

            metadata.insert(key, value);
        }

        Ok(Self {
            id,
            timestamp_ms,
            role,
            text,
            embedding,
            kv_cache,
            metadata,
        })
    }
}

/// Errors for attention state operations
#[derive(Debug, Clone)]
pub enum AttentionError {
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidFormat(String),
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionError::InvalidMagic => write!(f, "Invalid magic bytes"),
            AttentionError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            AttentionError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for AttentionError {}

/// A batch of attention states for efficient storage
#[derive(Debug, Clone)]
pub struct AttentionBatch {
    /// States in this batch
    pub states: Vec<AttentionState>,

    /// Session ID this batch belongs to
    pub session_id: Option<Id>,

    /// Document ID this batch belongs to
    pub document_id: Option<Id>,
}

impl AttentionBatch {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            session_id: None,
            document_id: None,
        }
    }

    pub fn with_session(mut self, session_id: Id) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_document(mut self, document_id: Id) -> Self {
        self.document_id = Some(document_id);
        self
    }

    pub fn add(&mut self, state: AttentionState) {
        self.states.push(state);
    }

    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.states.iter().map(|s| s.size_bytes()).sum()
    }

    /// Serialize batch to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic + version
        bytes.extend_from_slice(b"ATNB");
        bytes.extend_from_slice(&1u32.to_le_bytes());

        // Session ID
        if let Some(sid) = self.session_id {
            bytes.push(1);
            bytes.extend_from_slice(sid.as_bytes());
        } else {
            bytes.push(0);
        }

        // Document ID
        if let Some(did) = self.document_id {
            bytes.push(1);
            bytes.extend_from_slice(did.as_bytes());
        } else {
            bytes.push(0);
        }

        // States count
        bytes.extend_from_slice(&(self.states.len() as u32).to_le_bytes());

        // Each state
        for state in &self.states {
            let state_bytes = state.to_bytes();
            bytes.extend_from_slice(&(state_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&state_bytes);
        }

        bytes
    }

    /// Deserialize batch from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, AttentionError> {
        let mut offset = 0;

        // Magic
        if data.len() < 8 {
            return Err(AttentionError::InvalidFormat("Too short".into()));
        }
        if &data[0..4] != b"ATNB" {
            return Err(AttentionError::InvalidMagic);
        }
        offset += 4;

        // Version
        let version = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        if version != 1 {
            return Err(AttentionError::UnsupportedVersion(version));
        }
        offset += 4;

        // Session ID
        if data.len() < offset + 1 {
            return Err(AttentionError::InvalidFormat("Missing session flag".into()));
        }
        let has_session = data[offset] != 0;
        offset += 1;

        let session_id = if has_session {
            if data.len() < offset + 16 {
                return Err(AttentionError::InvalidFormat("Missing session ID".into()));
            }
            let mut id_bytes = [0u8; 16];
            id_bytes.copy_from_slice(&data[offset..offset + 16]);
            offset += 16;
            Some(Id::from_bytes(id_bytes))
        } else {
            None
        };

        // Document ID
        if data.len() < offset + 1 {
            return Err(AttentionError::InvalidFormat("Missing document flag".into()));
        }
        let has_document = data[offset] != 0;
        offset += 1;

        let document_id = if has_document {
            if data.len() < offset + 16 {
                return Err(AttentionError::InvalidFormat("Missing document ID".into()));
            }
            let mut id_bytes = [0u8; 16];
            id_bytes.copy_from_slice(&data[offset..offset + 16]);
            offset += 16;
            Some(Id::from_bytes(id_bytes))
        } else {
            None
        };

        // States count
        if data.len() < offset + 4 {
            return Err(AttentionError::InvalidFormat("Missing state count".into()));
        }
        let state_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        // States
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            if data.len() < offset + 8 {
                return Err(AttentionError::InvalidFormat("Missing state length".into()));
            }
            let state_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;

            if data.len() < offset + state_len {
                return Err(AttentionError::InvalidFormat("State truncated".into()));
            }
            let state = AttentionState::from_bytes(&data[offset..offset + state_len])?;
            offset += state_len;
            states.push(state);
        }

        Ok(Self {
            states,
            session_id,
            document_id,
        })
    }
}

impl Default for AttentionBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool, Role::Context] {
            let byte = role.to_byte();
            let restored = Role::from_byte(byte).unwrap();
            assert_eq!(role, restored);
        }
    }

    #[test]
    fn test_attention_state_roundtrip() {
        let state = AttentionState::new(
            Role::User,
            "Hello, how are you?".to_string(),
            vec![0.1, 0.2, 0.3, 0.4],
        )
        .with_metadata("turn", "1");

        let bytes = state.to_bytes();
        let restored = AttentionState::from_bytes(&bytes).unwrap();

        assert_eq!(state.role, restored.role);
        assert_eq!(state.text, restored.text);
        assert_eq!(state.embedding, restored.embedding);
        assert_eq!(state.metadata.get("turn"), restored.metadata.get("turn"));
    }

    #[test]
    fn test_attention_state_with_kv() {
        let kv = CompressedKV {
            model_id: "llama-3-8b".to_string(),
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            seq_len: 10,
            quantization: "fp16".to_string(),
            data: vec![1, 2, 3, 4, 5],
        };

        let state = AttentionState::new(
            Role::Assistant,
            "I'm doing well!".to_string(),
            vec![0.5, 0.6, 0.7, 0.8],
        )
        .with_kv_cache(kv);

        let bytes = state.to_bytes();
        let restored = AttentionState::from_bytes(&bytes).unwrap();

        assert!(restored.kv_cache.is_some());
        let restored_kv = restored.kv_cache.unwrap();
        assert_eq!(restored_kv.model_id, "llama-3-8b");
        assert_eq!(restored_kv.num_layers, 32);
        assert_eq!(restored_kv.data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_batch_roundtrip() {
        let mut batch = AttentionBatch::new()
            .with_session(Id::now());

        batch.add(AttentionState::new(
            Role::User,
            "Question 1".to_string(),
            vec![0.1, 0.2],
        ));

        batch.add(AttentionState::new(
            Role::Assistant,
            "Answer 1".to_string(),
            vec![0.3, 0.4],
        ));

        let bytes = batch.to_bytes();
        let restored = AttentionBatch::from_bytes(&bytes).unwrap();

        assert_eq!(restored.states.len(), 2);
        assert_eq!(restored.states[0].text, "Question 1");
        assert_eq!(restored.states[1].text, "Answer 1");
        assert!(restored.session_id.is_some());
    }
}
