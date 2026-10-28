CREATE DATABASE IF NOT EXISTS customer_analytics;
USE customer_analytics;

CREATE TABLE customers (
  customer_id VARCHAR(32) PRIMARY KEY,
  customer_code VARCHAR(64) NOT NULL UNIQUE,
  signup_date DATE NULL
);

CREATE TABLE orders (
  order_id VARCHAR(32) PRIMARY KEY,
  customer_id VARCHAR(32) NOT NULL,
  order_date DATE NOT NULL,
  order_status VARCHAR(20) NOT NULL,
  CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id)
    REFERENCES customers(customer_id),
  INDEX idx_orders_customer_date (customer_id, order_date)
);

CREATE TABLE order_items (
  item_id VARCHAR(32) PRIMARY KEY,
  order_id VARCHAR(32) NOT NULL,
  quantity INT NOT NULL,
  unit_price DECIMAL(12,2) NOT NULL,
  CONSTRAINT fk_items_order FOREIGN KEY (order_id)
    REFERENCES orders(order_id),
  CHECK (quantity > 0),
  CHECK (unit_price >= 0)
);

CREATE TABLE segment_runs (
  run_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  method VARCHAR(32) NOT NULL,
  run_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  parameters JSON NULL
);

CREATE TABLE customer_segments (
  run_id BIGINT NOT NULL,
  customer_id VARCHAR(32) NOT NULL,
  segment_label VARCHAR(80) NOT NULL,
  PRIMARY KEY (run_id, customer_id),
  CONSTRAINT fk_segments_run FOREIGN KEY (run_id)
    REFERENCES segment_runs(run_id),
  CONSTRAINT fk_segments_customer FOREIGN KEY (customer_id)
    REFERENCES customers(customer_id)
);
