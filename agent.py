from langchain.agents import initialize_agent, tool
from langchain.agents.agent_types import AgentType
from vector_stores import gemini_llm
from embedding import retrieve_doc

import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_system_prompt="""
You are a specialized data analysis agent designed to help users analyze CSV datasets using pandas DataFrames. Your role is to provide accurate, insightful analysis while being methodical and precise in your approach.

Your Capabilities
You have access to the following tools for DataFrame analysis:

load_csv: Load CSV files into memory as DataFrames
get_columns: Retrieve column names from the loaded DataFrame
describe_csv: Generate descriptive statistics for all numeric columns
average_col: Calculate the mean value of a specific column
st_dev_col: Calculate the standard deviation of a specific column
intelligent_plotter: Generates the most appropriate plot for one or two data columns to perform useful analysis.

Operating Guidelines
1. Always Load Data First
   Before performing any analysis, ensure the CSV file is loaded using the load_csv tool.
   If a user asks about data without specifying a file, ask them to provide the CSV file path.
   Handle file paths that may contain spaces (they will be converted automatically).

2. Understand Before Analyzing
   Use get_columns to understand the structure of the dataset before performing specific analyses.
   Use describe_csv to get an overview of the data distribution and basic statistics.
   This context will help you provide more meaningful insights.

3. Be Methodical in Analysis
   Break down complex questions into smaller, manageable steps.
   Use the appropriate tool for each specific task.
   Combine results from multiple tools when needed to provide comprehensive answers.

4. Provide Context and Interpretation
   Don't just report raw numbers - explain what they mean.
   Compare values when relevant (e.g., "Column A has a much higher standard deviation than Column B, indicating more variability").
   Identify potential data quality issues or interesting patterns.
   Suggest follow-up analyses when appropriate.

5. Handle Errors Gracefully
   If a column doesn't exist, check available columns first using get_columns.
   Provide helpful error messages and suggest corrections.
   If data isn't loaded, guide the user to load it first.

6. Communication Style
   Be extensive and clear in your explanations.
   Use proper statistical terminology but explain it when necessary.
   Structure your responses logically (overview → specific findings → conclusions).
   Ask clarifying questions when the user's request is ambiguous.

**VERY IMPORTANT: Output Format**
When you need to use a tool, you MUST respond in the following format:
Action: [the name of the tool to use, e.g., load_csv]
Action Input: [the input to the tool]

After the tool has been executed, you will receive an Observation. You will then think about the next step.
If you have gathered enough information to answer the user's question, you MUST provide the final answer in the following format:
Final Answer: [your comprehensive answer to the user's original question based on the tool outputs]

DO NOT output any other text after "Final Answer:". Your entire response should be just that line if you are providing the final answer.

Example Response Pattern
When analyzing data, follow this structure:
Data Loading: Confirm the dataset is loaded (internally, using tools).
Data Overview: Provide basic information about columns and structure (internally, using tools).
Specific Analysis: Perform the requested calculations (internally, using tools).
Final Output: Present the findings prefixed with "Final Answer:".

Important Notes
- Always verify data is loaded before attempting analysis.
- Column names are case-sensitive - use exact matches.
- Provide statistical context (e.g., what constitutes high/low variability).
- If asked about columns that don't exist, list available columns to help the user.
- Remember that you're working with a single DataFrame stored in memory.

Your goal is to make data analysis accessible and insightful, helping users understand their data through clear explanations and methodical analysis, strictly adhering to the specified output format.
"""

def Dataframe_agent(file_path,prompt,llm):
    df_store={}
    graphs=os.listdir("./Graphs")
    if len(graphs)!=0:
        for x in range(len(graphs)):
            os.remove(f"./Graphs/{graphs[x]}")

    @tool
    def load_csv(csv_path: str) -> str:
        '''Load a csv file and save it as a dataframe. This should be the first step.'''
        csv_path = csv_path.replace("_", " ")
        try:
            df = pd.read_csv(csv_path)
            # Store both the original and the working dataframe
            df_store["original_df"] = df.copy()
            df_store["df"] = df
            return f"File '{csv_path}' loaded successfully as a dataframe with {len(df)} rows and {len(df.columns)} columns."
        except Exception as e:
            return f"Error loading file: {str(e)}"

    @tool
    def get_columns(dummy: str) -> str:
        '''Returns the columns of the current dataframe. Input is ignored.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        try:
            columns = df_store['df'].columns.tolist()
            return f"Available columns: {columns}"
        except Exception as e:
            return f"Error getting columns: {str(e)}"
            
    # --- NEW: Data Inspection Tools ---
    
    @tool
    def show_head(dummy: str) -> str:
        '''Shows the first 5 rows of the current dataframe. Input is ignored.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        try:
            return f"First 5 rows of the dataframe:\n{df_store['df'].head().to_string()}"
        except Exception as e:
            return f"Error showing head: {str(e)}"

    @tool
    def get_info(dummy: str) -> str:
        '''Returns a technical summary of the dataframe (columns, data types, non-null counts). Input is ignored.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        try:
            # Capture the output of df.info() as it prints to stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                df_store['df'].info()
            info_str = buf.getvalue()
            return f"Dataframe Info:\n{info_str}"
        except Exception as e:
            return f"Error getting info: {str(e)}"
    
    # --- NEW: Filtering and State Management Tools ---

    @tool
    def filter_rows(query_str: str) -> str:
        '''
        Filters the dataframe based on a query string. The query must be in a format pandas.query() can understand (e.g., "Age > 30", "Country == 'USA'").
        This action modifies the current dataframe. Use reset_dataframe to undo.
        '''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        try:
            filtered_df = df_store['df'].query(query_str)
            df_store['df'] = filtered_df # Update the working dataframe
            if len(filtered_df) == 0:
                return f"Dataframe filtered with query '{query_str}'. No rows remaining."
            return f"Dataframe filtered with query '{query_str}'. {len(filtered_df)} rows remaining. Here are the first 5:\n{filtered_df.head().to_string()}"
        except Exception as e:
            return f"Error applying filter: {str(e)}. Please check your query syntax."
            
    @tool
    def reset_dataframe(dummy: str) -> str:
        '''Resets the dataframe to its original state, undoing any filtering. Input is ignored.'''
        if "original_df" not in df_store:
            return "Error: No original dataframe found. Please load a CSV file first."
        try:
            df_store["df"] = df_store["original_df"].copy()
            return f"Dataframe has been reset to its original state with {len(df_store['df'])} rows."
        except Exception as e:
            return f"Error resetting dataframe: {str(e)}"

    # --- Existing & NEW Statistical Tools ---

    @tool
    def describe_csv(dummy: str) -> str:
        '''Returns descriptive statistics for the numeric columns in the dataframe. Input is ignored.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        try:
            description = df_store['df'].describe()
            return f"Descriptive statistics:\n{description.to_string()}"
        except Exception as e:
            return f"Error describing dataframe: {str(e)}"

    @tool
    def calculate_correlation(dummy: str) -> str:
        '''Calculates and returns the correlation matrix for all numeric columns. Input is ignored.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        try:
            numeric_df = df_store['df'].select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                return "Error: Need at least two numeric columns to calculate correlation."
            corr_matrix = numeric_df.corr()
            return f"Correlation matrix:\n{corr_matrix.to_string()}"
        except Exception as e:
            return f"Error calculating correlation: {str(e)}"

    # --- Existing & NEW Plotting Tools ---
    
    @tool
    def plot_column(col: str) -> str:
        '''Creates a plot for a single column. It generates a histogram for numeric data and a pie chart for categorical data. 🥧'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        df = df_store['df']
        if col not in df.columns:
            return f"Error: Column '{col}' not found. Available columns: {df.columns.tolist()}"
        
        try:
            plt.figure(figsize=(10, 8))
            sns.set_theme(style="whitegrid")

            if pd.api.types.is_numeric_dtype(df[col]):
                # --- Histogram for Numeric Data ---
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                plot_type = 'histogram'
            else:
                # --- Pie Chart for Categorical Data ---
                all_counts = df[col].value_counts()
                top_counts = all_counts.nlargest(10)
                
                # If there are more than 10 categories, group the rest into 'Other'
                if len(all_counts) > 10:
                    other_sum = all_counts.iloc[10:].sum()
                    top_counts['Other'] = other_sum
                
                plt.pie(top_counts, labels=top_counts.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Proportion of Top 10 Categories in {col}')
                plt.ylabel('') # Hides default y-label for pie charts
                plt.axis('equal')  # Ensures the pie chart is a circle
                plot_type = 'piechart'

            save_path = f"./Graphs/{col.replace(' ', '_')}_{plot_type}.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            return f"Plot for column '{col}' was successfully saved to {save_path}"
        except Exception as e:
            return f"An error occurred while creating the plot: {str(e)}"
    
    @tool
    def plot_scatter(columns_str: str) -> str:
        '''Creates and saves a scatter plot between two numeric columns. Input must be 'column_x vs column_y'.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        
        try:
            parts = columns_str.split(' vs ')
            if len(parts) != 2:
                return "Error: Invalid format. Please use 'column_x vs column_y'."
            col_x, col_y = parts[0].strip(), parts[1].strip()

            df = df_store['df']
            if col_x not in df.columns or col_y not in df.columns:
                return f"Error: One or both columns not found. Available columns: {df.columns.tolist()}"
            if not pd.api.types.is_numeric_dtype(df[col_x]) or not pd.api.types.is_numeric_dtype(df[col_y]):
                return f"Error: Both columns must be numeric for a scatter plot."

            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            sns.scatterplot(data=df, x=col_x, y=col_y)
            plt.title(f'Scatter Plot of {col_x} vs {col_y}')
            
            save_path = f"./Graphs/{col_x}_vs_{col_y}_scatter.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            return f"Scatter plot for '{col_x}' vs '{col_y}' was successfully saved to {save_path}"
        except Exception as e:
            return f"An error occurred while creating the scatter plot: {str(e)}"
    @tool
    def get_value_percentages(col_name: str) -> str:
        '''Calculates and returns the percentage of each unique value in a specified column. Useful for categorical data.'''
        if "df" not in df_store:
            return "Error: No dataframe loaded. Please load a CSV file first."
        df = df_store['df']
        if col_name not in df.columns:
            return f"Error: Column '{col_name}' not found. Available columns: {df.columns.tolist()}"
        try:
            percentages = df[col_name].value_counts(normalize=True).mul(100).round(2)
            return f"Value percentages for column '{col_name}':\n{percentages.to_string()}"
        except Exception as e:
            return f"Error calculating value percentages for column '{col_name}': {str(e)}"

    tools = [load_csv, get_columns, show_head, get_info, filter_rows,describe_csv, reset_dataframe, calculate_correlation, plot_scatter,plot_column,get_value_percentages]
    agent=initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_error=True
    )
    full_prompt=f"{csv_system_prompt}\nThe user has asked: {prompt}\nThe designed file is {file_path}. Provide an extensive statistical analysis of the csv file. After you exposed your analysis to the user, plot a graph."
    return agent.run(full_prompt)

def PDFagent(prompt,prompt_embedding,collection,source,llm):
    context=retrieve_doc(prompt_embedding,collection,source,20)
    new_prompt=f"You are an expert assistant for question-answering tasks in Case-Aria s.r.l., a dairy company. Based on the provided context, provide a comprehensive yet clear answer to the question. Elaborate sufficiently on key concepts and processes to ensure a full understanding, directly addressing all parts of the question, but don't be too verbose. Question:\n{prompt}\n\nContext:\n{context}\nAnswer:"
    
    output=llm.invoke(new_prompt).content
    return output

