# It's exactly the same as chatbot-summarization.py, but with external memory.
# The external memory in the example is a SQLite database.
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# with ":memory:" we create an in-memory database
# conn = sqlite3.connect(":memory:", check_same_thread=False)

conn = sqlite3.connect("state_db/example.db") # We use a file-based database

memory = SqliteSaver(conn)

# builder = StateGraph(MessagesState)
# add nodes and edges
# compile the graph specifying the sqlite saver as the external memory
# graph = builder.compile(checkpointer=memory)
