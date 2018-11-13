DROP DATABASE if EXISTS labeled_coco;
CREATE DATABASE labeled_coco;
use labeled_coco;

DROP TABLE IF EXISTS Images;
CREATE TABLE Images (
   id INT NOT NULL AUTO_INCREMENT,
   path_url VARCHAR (100) NOT NULL,
   PRIMARY KEY (id)
);

DROP TABLE IF EXISTS Label;
CREATE TABLE Label (
   id INT NOT NULL AUTO_INCREMENT,
   name VARCHAR (80) NOT NULL,
   PRIMARY KEY (id)
);


DROP TABLE IF EXISTS Classified;
CREATE TABLE Classified(
    id_image INT NOT NULL,
    id_label INT NOT NULL,
    score FLOAT NOT NULL,
    FOREIGN KEY(id_image) REFERENCES Images(id),
    FOREIGN KEY(id_label) REFERENCES Label(id),
    PRIMARY KEY (id_image, id_label)
);
