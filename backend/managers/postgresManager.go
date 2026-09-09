package managers

import (
	"fmt"
	"os"
)

type PostgresManager struct {
	connString string
}

func NewPostgresManager() *PostgresManager {
	connString := os.Getenv("POSTGRES_URL")
	if connString == "" {
		fmt.Println("POSTGRES_URL not set, using default")
		connString = "postgresql://postgres:fake-pass@host:5432/postgres"
	}

	return &PostgresManager{
		connString: connString,
	}
}
