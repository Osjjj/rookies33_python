-- This is a SQL comment
SELECT * FROM users;

# Python style comment (should also be detected)

INSERT INTO users (id, name, email) VALUES (1, 'john', 'john@example.com');

UPDATE users SET name = 'kim' WHERE id = 1;

DELeTE FROM users WHERE id = 2;

/* Multi-line comment start
still comment line
end of comment */

DROP TABLE accounts;

CREATE TABLE test (
    id INT,
    name VARCHAR(100)
);

ALTER TABLE test ADD COLUMN age INT;

random text line

// JavaScript style comment

SELECT email FROM customers WHERE email LIKE '%@gmail.com';