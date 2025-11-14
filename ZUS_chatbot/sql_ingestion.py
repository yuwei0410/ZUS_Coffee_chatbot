import json
import re
import sqlite3
from datetime import datetime


def create_tables(cursor):
    """Creates the necessary tables in the database."""
    # Create the main outlets table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS outlets (
        outlet_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        address TEXT,
        maps_url VARCHAR(500)
    );
    """
    )

    # Create the detailed, queryable opening hours table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS opening_hours (
        hour_id INTEGER PRIMARY KEY AUTOINCREMENT,
        outlet_id INTEGER,
        day_of_week VARCHAR(10),
        open_time TIME,
        close_time TIME,
        FOREIGN KEY (outlet_id) REFERENCES outlets(outlet_id)
    );
    """
    )
    print("Tables created successfully (if they didn't exist).")


#
# REPLACE THIS FUNCTION IN code2
#
def parse_hour_string(hour_string):
    """
    Parses complex strings like "Monday: 9:00 AM – 1:00 PM, 2:00 – 6:00 PM"
    or "Saturday: Closed".
    Returns a LIST of tuples: [(day, open_time_24hr, close_time_24hr), ...]
    """
    try:
        day_name, hours_part = hour_string.split(":", 1)
        day_name = day_name.strip()
        hours_part = hours_part.strip()

        if "closed" in hours_part.lower():
            # This day is closed, return no entries
            return []

        # Split by comma for multiple time ranges (e.g., "9-1, 2-6")
        time_ranges_str_list = hours_part.split(",")

        results = []

        # Regex to find time ranges like "9:00 AM – 1:00 PM"
        # Handles different dashes [–-] and spacing \s*
        range_pattern = re.compile(r"([\d:]+\s*[AP]M)\s*[–-]\s*([\d:]+\s*[AP]M)")

        for range_str in time_ranges_str_list:
            match = range_pattern.search(range_str)
            if match:
                open_str_12hr, close_str_12hr = match.groups()

                # Convert to 24-hour format
                open_t = datetime.strptime(open_str_12hr.strip(), "%I:%M %p")
                close_t = datetime.strptime(close_str_12hr.strip(), "%I:%M %p")

                open_time_24hr = open_t.strftime("%H:%M:00")
                close_time_24hr = close_t.strftime("%H:%M:00")

                results.append((day_name, open_time_24hr, close_time_24hr))

        return results

    except Exception as e:
        print(f"Warning: Could not parse string '{hour_string}'. Error: {e}")
        return []


#
# REPLACE THIS FUNCTION IN code2
#
def process_json_data(cursor, data):
    """Inserts the JSON data into the database."""

    # 1. Insert into the 'outlets' table
    cursor.execute(
        """
    INSERT INTO outlets (name, address, maps_url)
    VALUES (?, ?, ?)
    """,
        (data["name"], data["address"], data["maps_url"]),
    )

    # 2. Get the new 'outlet_id' that was just created
    new_outlet_id = cursor.lastrowid
    print(f"Inserted '{data['name']}' with outlet_id: {new_outlet_id}")

    # 3. Loop and insert all opening hours
    hours_to_insert = []
    for hour_str in data["opening_hours"]:
        # THIS IS THE CHANGED PART:
        # parse_hour_string now returns a LIST of entries
        parsed_entries = parse_hour_string(hour_str)

        # Loop through the list (e.g., for morning and afternoon shifts)
        for entry in parsed_entries:
            day, open_time, close_time = entry
            hours_to_insert.append((new_outlet_id, day, open_time, close_time))

    # 4. Use 'executemany' for efficient bulk insertion
    cursor.executemany(
        """
    INSERT INTO opening_hours (outlet_id, day_of_week, open_time, close_time)
    VALUES (?, ?, ?, ?)
    """,
        hours_to_insert,
    )

    print(f"Inserted {len(hours_to_insert)} opening hour entries.")


# --- Main execution ---
if __name__ == "__main__":

    DB_FILE = "outlets.db"
    JSON_FILE = "zus_outlets.json"

    # Connect to the SQLite database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # Set up the database tables
        create_tables(cursor)

        # Load the data from the JSON file
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        # --- THIS IS THE CORRECTED LOGIC ---

        if isinstance(loaded_data, dict):
            # The file is a single JSON object
            print("Processing 1 outlet from JSON object...")
            process_json_data(cursor, loaded_data)

        elif isinstance(loaded_data, list):
            # The file is a list of JSON objects
            print(f"Processing {len(loaded_data)} outlets from JSON list...")
            for outlet_object in loaded_data:
                process_json_data(cursor, outlet_object)
        else:
            print("Error: JSON root is not a dictionary or a list.")

        # --- END OF CORRECTION ---

        # Commit (save) all changes to the database
        conn.commit()
        print("\nAll data successfully imported and saved!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        conn.rollback()  # Undo any changes if an error happened

    finally:
        # Always close the connection
        conn.close()
