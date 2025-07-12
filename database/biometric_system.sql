CREATE DATABASE IF NOT EXISTS biometric_system
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Veritabanını kullan
USE biometric_system;

-- Kişiler tablosu
CREATE TABLE IF NOT EXISTS persons (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT 'Kişinin adı soyadı',
    employee_id VARCHAR(50) UNIQUE NOT NULL COMMENT 'Çalışan kimlik numarası',
    department VARCHAR(100) DEFAULT NULL COMMENT 'Çalıştığı departman',
    phone VARCHAR(20) DEFAULT NULL COMMENT 'Telefon numarası',
    email VARCHAR(100) DEFAULT NULL COMMENT 'E-posta adresi',
    face_image LONGBLOB DEFAULT NULL COMMENT 'Yüz fotoğrafı (Base64)',
    face_embeddings JSON DEFAULT NULL COMMENT 'Yüz embedding vektörleri (JSON array)',
    embedding_count INT DEFAULT 0 COMMENT 'Kaydedilmiş embedding sayısı',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Kayıt tarihi',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Güncelleme tarihi',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Aktif durumu',
    
    INDEX idx_employee_id (employee_id),
    INDEX idx_name (name),
    INDEX idx_department (department),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB COMMENT='Kayıtlı kişiler tablosu';

-- Geçiş kayıtları tablosu
CREATE TABLE IF NOT EXISTS access_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person_id INT NOT NULL COMMENT 'Kişi ID referansı',
    access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Geçiş zamanı',
    access_type ENUM('entry', 'exit', 'internal') DEFAULT 'entry' COMMENT 'Geçiş türü: giriş/çıkış/iç kapı',
    confidence_score DECIMAL(4,3) DEFAULT NULL COMMENT 'Tanıma güven skoru (0.000-1.000)',
    location VARCHAR(100) DEFAULT 'Ana Giriş' COMMENT 'Geçiş lokasyonu',
    device_info VARCHAR(200) DEFAULT NULL COMMENT 'Cihaz bilgisi',
    notes TEXT DEFAULT NULL COMMENT 'Ek notlar',
    
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    INDEX idx_person_id (person_id),
    INDEX idx_access_time (access_time),
    INDEX idx_access_type (access_type),
    INDEX idx_location (location)
) ENGINE=InnoDB COMMENT='Geçiş kayıtları tablosu';

-- Mevcut veritabanlarını güncellemek için (eğer tablo zaten varsa)
ALTER TABLE access_logs MODIFY COLUMN access_type ENUM('entry', 'exit', 'internal') DEFAULT 'entry' COMMENT 'Geçiş türü: giriş/çıkış/iç kapı';