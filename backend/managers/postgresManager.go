package managers

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"

	"github.com/golang-migrate/migrate/v4"
	pgxmigrate "github.com/golang-migrate/migrate/v4/database/pgx/v5"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"github.com/jackc/pgx/v5/pgxpool"
	_ "github.com/jackc/pgx/v5/stdlib"

	"daybid-dev-service/migrations"
)

type PostgresManager struct {
	Pool *pgxpool.Pool
}

// NewPostgresManager connects to Postgres, running any pending migrations
// first so the schema is always up to date before the pool is handed out.
func NewPostgresManager() *PostgresManager {
	connString := postgresConnString()

	if err := runMigrations(connString); err != nil {
		log.Fatalf("postgres: migrations failed: %v", err)
	}

	pool, err := pgxpool.New(context.Background(), connString)
	if err != nil {
		log.Fatalf("postgres: failed to create connection pool: %v", err)
	}

	if err := pool.Ping(context.Background()); err != nil {
		log.Fatalf("postgres: ping failed: %v", err)
	}

	return &PostgresManager{Pool: pool}
}

// postgresConnString builds the DSN from the individual POSTGRES_* variables
// rather than trusting a pre-built POSTGRES_URL, so the app doesn't depend on
// ${VAR}-style interpolation inside an env_file being supported.
func postgresConnString() string {
	user := envOrDefault("POSTGRES_USER", "postgres")
	password := envOrDefault("POSTGRES_PASSWORD", "fake-pass")
	host := envOrDefault("POSTGRES_HOST", "localhost")
	port := envOrDefault("POSTGRES_PORT", "5432")
	database := envOrDefault("POSTGRES_DB", "postgres")

	return fmt.Sprintf("postgresql://%s:%s@%s:%s/%s", user, password, host, port, database)
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func runMigrations(connString string) error {
	db, err := sql.Open("pgx/v5", connString)
	if err != nil {
		return fmt.Errorf("open migration connection: %w", err)
	}
	defer db.Close()

	driver, err := pgxmigrate.WithInstance(db, &pgxmigrate.Config{})
	if err != nil {
		return fmt.Errorf("init migrate driver: %w", err)
	}

	source, err := iofs.New(migrations.FS, ".")
	if err != nil {
		return fmt.Errorf("open migrations source: %w", err)
	}

	m, err := migrate.NewWithInstance("iofs", source, "pgx5", driver)
	if err != nil {
		return fmt.Errorf("init migrate: %w", err)
	}
	defer m.Close()

	if err := m.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("run migrations: %w", err)
	}

	return nil
}
