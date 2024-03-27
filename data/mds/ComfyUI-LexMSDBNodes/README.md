# MSSQL Interaction Utility

This repository contains a set of Python classes designed to interact with an MSSQL server. These classes facilitate operations such as connecting to the server, reading configuration details, querying the database, and performing CRUD operations on database tables.

Please note that this is still in progress and extended functionailty will be added all the time.
Currently the table is hardcoded in the MSSQLNode.py folder and its set to txt2img you can change that table as much as you like and it will detect the fields on reload
![image](https://github.com/SOELexicon/ComfyUI-LexMSDBNodes/assets/4205001/737b51bf-5d11-4080-8795-6611da4c523f)


## Key Features

- **Flexibility**: 	A wide range of SQL operations can be performed, from basic queries to more complex insert and update operations.
			The Nodes inputs will automatically show based on the collumns and those collumns types in the table
- **Usability**: The project includes classes to manage database connection and perform specific tasks such as executing a query or managing table records.
- **Compatibility**: It incorporates additional libraries like PyODBC, NumPy, PyTorch, and OpenCV, bridging the gap between database management and data science.
- **Modularity**: The project is designed to be modular, with different classes handling different aspects of the database interaction.

## Components

- `MSSQLFn class`: A utility class with several static methods for connecting to the database, reading configuration information, and loading table metadata.
- `MSSQLQueryNode, MSSqlNode, MSSqlTableNode, and MSSqlSelectNode classes`: These classes each encapsulate a specific functionality related to SQL operations. These range from executing specific queries to performing more complex operations such as insert or update records in a table.
- `Node class mappings`: A dictionary mapping string names to the respective classes, useful for dynamically instantiating the necessary class based on the input or operation required.

## Installation

1. Clone this repository or download the code.
2. Ensure you have Python installed on your machine (preferably Python 3.8 or newer).
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Ensure your MSSQL server is up and running and accessible from the machine where this code will be executed.

## Table Example
/****** Object:  Table [dbo].[txt2img]    Script Date: 20/07/2023 21:33:13 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

````
CREATE TABLE [dbo].[txt2img](
	[id] [bigint] IDENTITY(1,1) NOT NULL,
	[DateAdded] [datetime] NULL,
	[Image] [varbinary](max) NULL,
	[seed] [bigint] NULL,
	[subseed] [bigint] NULL,
	[subseed_strength] [decimal](18, 0) NULL,
	[sampler] [nvarchar](50) NULL,
	[batch_size] [int] NULL,
	[n_iter] [int] NULL,
	[steps] [int] NULL,
	[cfg_scale] [int] NULL,
	[clip_skip] [int] NULL,
	[width] [int] NULL,
	[height] [int] NULL,
	[restore_faces] [bit] NULL,
	[positive_prompt] [nvarchar](4000) NULL,
	[negative_prompt] [nvarchar](4000) NULL,
	[sampler_index] [nvarchar](50) NULL,
 CONSTRAINT [PK_txt2img] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
````




## Usage

To use this repository, import the necessary classes into your Python script and instantiate them as needed. Each class has specific methods to perform different operations, which can be used as per the requirements of your project.

Here is a general example of what the ComfyUI\custom_nodes\ComfyUI-LexMSDBNodes\nodes\config.ini file might look like:
````
[MSSQL]
server = your_server
database = your_database
username = your_username
password = your_password
driver = {SQL Server Native Client 11.0}
integrated_security = False
````
## Contributing

Contributions to the project are welcome. To contribute:
1. Fork the repository.
2. Make your changes or additions in your forked repository.
3. Create a pull request detailing the changes you made.


## License

This project is open source and is licensed under the [MIT License](LICENSE).
