//! # Blob
//!
//! Raw payload data attached to a point.
//!
//! ARMS doesn't interpret this data - it's yours.
//! Could be: tensor bytes, text, compressed state, anything.
//!
//! Separation of concerns:
//! - Point = WHERE (position in space)
//! - Blob = WHAT (the actual data)

/// Raw data attached to a point
///
/// ARMS stores this opaquely. You define what it means.
#[derive(Clone, Debug, PartialEq)]
pub struct Blob {
    data: Vec<u8>,
}

impl Blob {
    /// Create a new blob from bytes
    ///
    /// # Example
    /// ```
    /// use arms::Blob;
    /// let blob = Blob::new(vec![1, 2, 3, 4]);
    /// assert_eq!(blob.size(), 4);
    /// ```
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Create an empty blob
    ///
    /// Useful when you only care about position, not payload.
    pub fn empty() -> Self {
        Self { data: vec![] }
    }

    /// Create a blob from a string (UTF-8 bytes)
    ///
    /// # Example
    /// ```
    /// use arms::Blob;
    /// let blob = Blob::from_str("hello");
    /// assert_eq!(blob.as_str(), Some("hello"));
    /// ```
    pub fn from_str(s: &str) -> Self {
        Self {
            data: s.as_bytes().to_vec(),
        }
    }

    /// Get the raw bytes
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if the blob is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Try to interpret as UTF-8 string
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }

    /// Consume and return the inner data
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

impl From<Vec<u8>> for Blob {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

impl From<&[u8]> for Blob {
    fn from(data: &[u8]) -> Self {
        Self::new(data.to_vec())
    }
}

impl From<&str> for Blob {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

impl From<String> for Blob {
    fn from(s: String) -> Self {
        Self::new(s.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_new() {
        let blob = Blob::new(vec![1, 2, 3]);
        assert_eq!(blob.data(), &[1, 2, 3]);
        assert_eq!(blob.size(), 3);
    }

    #[test]
    fn test_blob_empty() {
        let blob = Blob::empty();
        assert!(blob.is_empty());
        assert_eq!(blob.size(), 0);
    }

    #[test]
    fn test_blob_from_str() {
        let blob = Blob::from_str("hello world");
        assert_eq!(blob.as_str(), Some("hello world"));
    }

    #[test]
    fn test_blob_as_str_invalid_utf8() {
        let blob = Blob::new(vec![0xff, 0xfe]);
        assert_eq!(blob.as_str(), None);
    }

    #[test]
    fn test_blob_from_conversions() {
        let blob1: Blob = vec![1, 2, 3].into();
        assert_eq!(blob1.size(), 3);

        let blob2: Blob = "test".into();
        assert_eq!(blob2.as_str(), Some("test"));

        let blob3: Blob = String::from("test").into();
        assert_eq!(blob3.as_str(), Some("test"));
    }

    #[test]
    fn test_blob_into_inner() {
        let blob = Blob::new(vec![1, 2, 3]);
        let data = blob.into_inner();
        assert_eq!(data, vec![1, 2, 3]);
    }
}
