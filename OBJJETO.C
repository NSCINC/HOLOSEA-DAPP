#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>
#include <string.h>

// Function prototypes
void initializeDatabase();
void readCSVData(const char *filePath);
void insertInvestor(const char *name, const char *email, const char *phone);
int executeSQL(const char *sql);

int main() {
    initializeDatabase();
    readCSVData("investors_data.csv");
    return 0;
}

// Initialize the database and create tables
void initializeDatabase() {
    const char *createInvestorsTable = 
        "CREATE TABLE IF NOT EXISTS Investors ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, "
        "email TEXT NOT NULL, "
        "phone_number TEXT NOT NULL);";

    const char *createAssetsTable = 
        "CREATE TABLE IF NOT EXISTS Assets ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "asset_name TEXT NOT NULL, "
        "asset_type TEXT NOT NULL, "
        "total_slots INTEGER NOT NULL, "
        "annual_return REAL NOT NULL);";

    const char *createInvestmentsTable = 
        "CREATE TABLE IF NOT EXISTS Investments ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "investor_id INTEGER, "
        "asset_id INTEGER, "
        "invested_amount REAL, "
        "FOREIGN KEY(investor_id) REFERENCES Investors(id), "
        "FOREIGN KEY(asset_id) REFERENCES Assets(id));";

    // Execute SQL statements
    executeSQL(createInvestorsTable);
    executeSQL(createAssetsTable);
    executeSQL(createInvestmentsTable);
}

// Function to execute SQL commands
int executeSQL(const char *sql) {
    sqlite3 *db;
    char *errorMessage = 0;

    // Open the database
    int rc = sqlite3_open("investments.db", &db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return rc;
    }

    // Execute SQL statement
    rc = sqlite3_exec(db, sql, 0, 0, &errorMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errorMessage);
        sqlite3_free(errorMessage);
    } else {
        printf("SQL executed successfully.\n");
    }

    // Close the database
    sqlite3_close(db);
    return rc;
}

// Read investor data from a CSV file and insert into database
void readCSVData(const char *filePath) {
    FILE *file = fopen(filePath, "r");
    if (!file) {
        perror("Could not open file");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        char name[100], email[100], phone[20];
        // Use sscanf to parse the CSV line
        if (sscanf(line, "%99[^,],%99[^,],%19[^\n]", name, email, phone) == 3) {
            printf("Name: %s, Email: %s, Phone: %s\n", name, email, phone);
            insertInvestor(name, email, phone); // Insert data into the database
        }
    }

    fclose(file);
}

// Insert investor data into the database
void insertInvestor(const char *name, const char *email, const char *phone) {
    sqlite3 *db;
    sqlite3_stmt *stmt;

    // Open the database
    int rc = sqlite3_open("investments.db", &db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return;
    }

    const char *insertInvestorSQL = "INSERT INTO Investors (name, email, phone_number) VALUES (?, ?, ?)";
    
    // Prepare the SQL statement
    rc = sqlite3_prepare_v2(db, insertInvestorSQL, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
    } else {
        // Bind values
        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, email, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, phone, -1, SQLITE_STATIC);

        // Execute the statement
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Execution failed: %s\n", sqlite3_errmsg(db));
        } else {
            printf("Investor inserted successfully.\n");
        }
    }

    // Finalize and close the statement and database
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}
