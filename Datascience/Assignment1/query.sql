use mysql;

create table if not exists score(
	student_number	int		primary key,
	attendance	decimal(3,2),
	homework	decimal(4,2),
	discussion	int,
	midterm		decimal(4,2),
	final		decimal(4,2),
	score		decimal(4,2),
	grade 		char(1)
);
